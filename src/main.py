# =============================
# FILE: src/main.py
# =============================
import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_THREADS = (os.cpu_count() or 4) * 2
from .pdf.extract import PDFExtractor
from .steps.section_labeler import label_section
from .steps.generic_section_labeler import label_generic_section
from .steps.per_paper_extract import extract_paper
from .steps.generic_document_extract import extract_generic_document
from .steps.content_type_detector import detect_content_type
from .steps.cluster_topics import cluster_topics
from .steps.normalize_techniques import normalize_techniques
from .graph.build import build_graph_and_write


def process_single_pdf(args):
    """Process a single PDF file - designed for threading"""
    idx, pdf, extractor = args
    pdf_id = f"P{idx+1}"
    
    # Extract text and chunk
    pages = extractor.extract_pages(pdf)
    chunks = extractor.chunkify(pdf_id, pages)
    
    if not chunks:
        print(f"Warning: No text extracted from {pdf.stem}")
        return None
    
    # Detect content type using first few chunks
    sample_content = " ".join([ch["text"] for ch in chunks[:3]])
    content_type = detect_content_type(sample_content)
    
    print(f"Processing {pdf.stem}: {content_type.content_type} (confidence: {content_type.confidence:.2f})")
    
    # Label sections based on content type
    chunks_labeled = []
    if content_type.content_type == "scientific_paper":
        # Use scientific section labeling
        for ch in chunks:
            lab = label_section(ch["text"])
            chunks_labeled.append({**ch, "section": lab.section, "confidence": lab.confidence})
        
        # Extract using scientific paper approach
        per_paper = extract_paper(pdf_id, chunks_labeled)
        topics = per_paper.topics
        techniques = per_paper.techniques
        per_paper_data = json.loads(per_paper.model_dump_json())
        
    else:
        # Use generic section labeling
        for ch in chunks:
            lab = label_generic_section(ch["text"])
            chunks_labeled.append({**ch, "section": lab.section, "confidence": lab.confidence})
        
        # Extract using generic document approach
        generic_extraction = extract_generic_document(pdf_id, chunks_labeled)
        
        # Convert generic extraction to match expected format
        topics = []
        techniques = []
        per_paper_data = {
            "pdf_id": pdf_id,
            "topics": [],
            "techniques": []
        }
        
        # Convert generic topics to paper topics format
        for topic in generic_extraction.topics:
            paper_topic = {
                "id": topic.id,
                "label": topic.label,
                "evidence_chunks": [{"chunk_id": ev.chunk_id, "rationale": ev.rationale} for ev in topic.evidence_chunks]
            }
            topics.append(paper_topic)
            per_paper_data["topics"].append(paper_topic)
        
        # Convert generic concepts to techniques format
        for concept in generic_extraction.concepts:
            technique = {
                "canonical": concept.canonical,
                "variants": concept.variants,
                "evidence_chunks": [{"chunk_id": ev.chunk_id, "rationale": ev.rationale} for ev in concept.evidence_chunks]
            }
            techniques.append(technique)
            per_paper_data["techniques"].append(technique)
    
    # Extract paper content for graph building
    paper_content = [ch["text"] for ch in chunks_labeled]
    
    return {
        "pdf_id": pdf_id,
        "title": pdf.stem,
        "content_type": content_type.content_type,
        "content_type_confidence": content_type.confidence,
        "per_paper": per_paper_data,
        "topics": topics,
        "techniques": techniques,
        "content": paper_content
    }


def run(pdf_dir: Path, out_dir: Path, topics: int, chunk_words: int, chunk_overlap: int, threads: int = DEFAULT_THREADS):
    out_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted([p for p in Path(pdf_dir).glob("**/*.pdf")])
    if not pdfs:
        raise SystemExit(f"No PDFs found in {pdf_dir}")

    extractor = PDFExtractor(chunk_words=chunk_words, overlap_words=chunk_overlap)

    per_paper_results = []
    paper_rows = []
    paper_level_topics_flat = []
    technique_candidates = set()
    papers_content = {}  # Store paper content for graph building

    # Prepare arguments for threading
    pdf_args = [(idx, pdf, extractor) for idx, pdf in enumerate(pdfs)]
    
    # Process PDFs in parallel
    print(f"Processing {len(pdfs)} PDFs using {threads} threads...")
    with ThreadPoolExecutor(max_workers=threads) as executor:
        # Submit all tasks
        future_to_pdf = {executor.submit(process_single_pdf, args): args for args in pdf_args}
        
        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_pdf), total=len(pdfs), desc="Processing PDFs"):
            try:
                result = future.result()
                
                if result is None:
                    continue  # Skip failed extractions
                
                # Aggregate results
                per_paper_results.append(result["per_paper"])
                paper_rows.append({
                    "pdf_id": result["pdf_id"], 
                    "title": result["title"],
                    "content_type": result.get("content_type", "unknown"),
                    "content_type_confidence": result.get("content_type_confidence", 0.0)
                })
                papers_content[result["pdf_id"]] = result["content"]
                
                # Aggregate for corpus clustering
                for t in result["topics"]:
                    # Handle both dict and object formats
                    if isinstance(t, dict):
                        paper_level_topics_flat.append({"id": t["id"], "label": t["label"]})
                    else:
                        paper_level_topics_flat.append({"id": t.id, "label": t.label})
                
                # Technique candidates
                for te in result["techniques"]:
                    # Handle both dict and object formats
                    if isinstance(te, dict):
                        technique_candidates.add(te["canonical"])
                        for v in te.get("variants", []):
                            technique_candidates.add(v)
                    else:
                        technique_candidates.add(te.canonical)
                        for v in te.variants:
                            technique_candidates.add(v)
                        
            except Exception as e:
                pdf_idx, pdf_path, _ = future_to_pdf[future]
                print(f"Error processing PDF {pdf_path}: {e}")
                # Continue with other PDFs

    # cluster corpus topics
    corpus = cluster_topics(paper_level_topics_flat, k=topics)
    corpus_dict = json.loads(corpus.model_dump_json())

    # normalize techniques
    norm = normalize_techniques(sorted(list(technique_candidates)))
    norm_dict = json.loads(norm.model_dump_json())

    # write graph & viewer
    papers_df = pd.DataFrame(paper_rows)
    build_graph_and_write(papers_df, per_paper_results, corpus_dict, norm_dict, str(out_dir), papers_content, threads)


def main():
    ap = argparse.ArgumentParser(description="ChatGPT‑only PDF → Mind‑Map pipeline")
    ap.add_argument("--pdf_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("./output"))
    ap.add_argument("--topics", type=int, default=12)
    ap.add_argument("--chunk_words", type=int, default=900)
    ap.add_argument("--chunk_overlap", type=int, default=120)
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="Number of concurrent threads (default: 2x CPU count)")
    args = ap.parse_args()
    run(args.pdf_dir, args.out_dir, args.topics, args.chunk_words, args.chunk_overlap, args.threads)

if __name__ == "__main__":
    main()
