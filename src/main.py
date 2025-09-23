# =============================
# FILE: src/main.py
# =============================
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from .pdf.extract import PDFExtractor
from .steps.section_labeler import label_section
from .steps.per_paper_extract import extract_paper
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
    
    # Label sections via LLM
    chunks_labeled = []
    for ch in chunks:
        lab = label_section(ch["text"])
        chunks_labeled.append({**ch, "section": lab.section, "confidence": lab.confidence})
    
    # Per-paper topics + techniques
    per_paper = extract_paper(pdf_id, chunks_labeled)
    
    # Extract paper content for graph building
    paper_content = [ch["text"] for ch in chunks_labeled]
    
    return {
        "pdf_id": pdf_id,
        "title": pdf.stem,
        "per_paper": json.loads(per_paper.model_dump_json()),
        "topics": per_paper.topics,
        "techniques": per_paper.techniques,
        "content": paper_content
    }


def run(pdf_dir: Path, out_dir: Path, topics: int, chunk_words: int, chunk_overlap: int, threads: int = 10):
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
                
                # Aggregate results
                per_paper_results.append(result["per_paper"])
                paper_rows.append({"pdf_id": result["pdf_id"], "title": result["title"]})
                papers_content[result["pdf_id"]] = result["content"]
                
                # Aggregate for corpus clustering
                for t in result["topics"]:
                    paper_level_topics_flat.append({"id": t.id, "label": t.label})
                
                # Technique candidates
                for te in result["techniques"]:
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
    build_graph_and_write(papers_df, per_paper_results, corpus_dict, norm_dict, str(out_dir), papers_content)


def main():
    ap = argparse.ArgumentParser(description="ChatGPT‑only PDF → Mind‑Map pipeline")
    ap.add_argument("--pdf_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("./output"))
    ap.add_argument("--topics", type=int, default=12)
    ap.add_argument("--chunk_words", type=int, default=900)
    ap.add_argument("--chunk_overlap", type=int, default=120)
    ap.add_argument("--threads", type=int, default=10, help="Number of concurrent threads for PDF processing")
    args = ap.parse_args()
    run(args.pdf_dir, args.out_dir, args.topics, args.chunk_words, args.chunk_overlap, args.threads)

if __name__ == "__main__":
    main()
