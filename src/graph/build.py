# =============================
# FILE: src/graph/build.py
# =============================
from typing import Dict, List, Any
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..llm.client import embed
from ..steps.summarize_topic import summarize_topic
from ..steps.relink_topics import find_relevant_papers_for_topic
from ..steps.enrich_techniques import enrich_technique
from ..steps.enrich_papers import enrich_paper
from ..steps.extract_ideas import extract_ideas_for_topic
from ..steps.merge_subtopics_advanced import merge_subtopics_advanced
from .markdown_export import export_to_markdown_outline

HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Paper Topics Mind‑Map</title>
  <script src=\"https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js\"></script>
  <style>
    html, body { height: 100%; margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
    #app { display: grid; grid-template-columns: 3fr 2fr; height: 100%; }
    #cy { border-right: 1px solid #eee; }
    #side { padding: 16px; overflow: auto; }
    .muted { color: #666; }
    .kv { margin: 8px 0; }
    .kv b { display: inline-block; min-width: 90px; }
  </style>
</head>
<body>
  <div id=\"app\">
    <div id=\"cy\"></div>
    <div id=\"side\">
      <h2>Paper Topics Mind-Map</h2>
      <p class=\"muted\">Click nodes to see details. Drag to rearrange. Scroll to zoom.</p>
      <div id=\"info\"></div>
    </div>
  </div>
  <script>
    const DATA = __GRAPH_JSON__;
    const cy = cytoscape({
      container: document.getElementById('cy'),
      elements: DATA.elements,
      layout: { 
        name: 'cose', 
        animate: false,
        nodeRepulsion: 400000,
        idealEdgeLength: 100,
        nodeOverlap: 20,
        gravity: 80,
        numIter: 1000,
        initialTemp: 200,
        coolingFactor: 0.95,
        minTemp: 1.0
      },
      style: [
        { selector: 'node[type="topic"]', style: { 'background-color': '#5B8DEF', 'label': 'data(label)', 'font-size': 5, 'width': 'mapData(size, 20, 100, 15, 80)', 'height': 'mapData(size, 20, 100, 15, 80)', 'text-wrap': 'wrap', 'text-max-width': 120 }},
        { selector: 'node[type="paper"]', style: { 'background-color': '#33B679', 'label': 'data(label)', 'font-size': 4, 'width': 'mapData(size, 12, 40, 10, 30)', 'height': 'mapData(size, 12, 40, 10, 30)', 'text-wrap': 'wrap', 'text-max-width': 120 }},
        { selector: 'node[type="technique"]', style: { 'background-color': '#F6BF26', 'shape': 'round-rectangle', 'label': 'data(label)', 'font-size': 5, 'width': 'label', 'padding': '5px' }},
        { selector: 'edge[rel="member"]', style: { 'line-color': '#999', 'width': 'mapData(weight, 0.05, 0.9, 0.5, 4)', 'target-arrow-shape': 'triangle', 'target-arrow-color': '#999', 'curve-style': 'bezier' }},
        { selector: 'edge[rel="mentions"]', style: { 'line-color': '#F6BF26', 'width': 'mapData(weight, 1, 20, 0.5, 4)', 'curve-style': 'bezier' }},
        { selector: 'edge[rel="similar"]', style: { 'line-color': '#5B8DEF', 'line-style': 'dashed', 'width': 'mapData(weight, 0.25, 1.0, 1, 5)', 'curve-style': 'bezier' }}
      ]
    });
    const info = document.getElementById('info');
    function renderDetails(ele){
      const d = ele.data();
      let html = `<div class=\"kv\"><b>ID</b> ${d.id}</div>` + `<div class=\"kv\"><b>Type</b> ${d.type || 'edge'}</div>`;
      if (d.type === 'topic'){ 
        html += `<div class=\"kv\"><b>Summary</b> ${d.summary || ''}</div>`; 
        if (d.subtopics && d.subtopics.length > 0) {
          html += `<div class=\"kv\"><b>Related Concepts</b><ul>${d.subtopics.map(s => `<li>${s}</li>`).join('')}</ul></div>`;
        }
        if (d.id === 'T_Unrelated' && d.unmatched_topics && d.unmatched_topics.length > 0) {
          html += `<div class=\"kv\"><b>Unmatched Topics</b><ul>${d.unmatched_topics.map(s => `<li>${s}</li>`).join('')}</ul></div>`;
        }
      }
      if (d.type === 'paper'){ 
        html += `<div class=\"kv\"><b>Title</b> ${d.title}</div>`; 
        if (d.summary) { html += `<div class=\"kv\"><b>Summary</b> ${d.summary}</div>`; }
        if (d.topics) { html += `<div class=\"kv\"><b>Topics</b> ${d.topics.join(', ')}</div>`; }
      }
      if (d.type === 'technique'){ 
        html += `<div class=\"kv\"><b>Technique</b> ${d.label}</div>`; 
        if (d.summary) { html += `<div class=\"kv\"><b>Summary</b> ${d.summary}</div>`; }
        if (d.how_it_works) { html += `<div class=\"kv\"><b>How it Works</b><ul>${d.how_it_works.map(s => `<li>${s}</li>`).join('')}</ul></div>`; }
        if (d.relevance) { html += `<div class=\"kv\"><b>Relevance</b><ul>${d.relevance.map(s => `<li>${s}</li>`).join('')}</ul></div>`; }
      }
      if (d.rel){ html += `<div class=\"kv\"><b>Relation</b> ${d.rel}</div>`; html += `<div class=\"kv\"><b>Weight</b> ${d.weight || ''}</div>`; }
      info.innerHTML = html;
    }
    cy.on('tap', 'node', evt => renderDetails(evt.target));
    cy.on('tap', 'edge', evt => renderDetails(evt.target));
  </script>
</body>
</html>
"""


def build_graph_and_write(
    papers_df: pd.DataFrame,
    per_paper: list[dict],
    corpus_topics: dict,
    norm_tech: dict,
    out_dir: str,
    papers_content: Dict[str, List[str]],
    threads: int = 10,
):
    # Build nodes
    nodes, edges = [], []

    # Papers
    for _, r in papers_df.iterrows():
        nodes.append({"data": {"id": r["pdf_id"], "label": r["pdf_id"], "title": r["title"], "type": "paper", "size": 18}})

    # Corpus topics
    topic_nodes = {}
    for t in corpus_topics.get("topics", []):
        tid = t["id"]
        nodes.append({"data": {"id": tid, "label": t["label"], "type": "topic", "size": 40}})
        topic_nodes[tid] = t

    # Techniques (candidates)
    tech_nodes_to_add = []
    for g in norm_tech.get("groups", []):
        nid = f"Tech:{slug(g['canonical'])}"
        tech_nodes_to_add.append({"data": {"id": nid, "label": g["canonical"], "type": "technique", "size": 22}})

    # New pass: Advanced multi-pass merging with LLM judgment
    subtopic_mapping = merge_subtopics_advanced(per_paper, corpus_topics.get("topics", []), threads=threads)
    
    # paper → topic edges via LLM membership scores (augmented with subtopic mapping)
    connected_paper_ids = set()
    member_map = {m: t["id"] for t in corpus_topics.get("topics", []) for m in t.get("member_topic_ids", [])}
    for sub_id, main_id in subtopic_mapping.items():
        member_map[sub_id] = main_id

    for p in per_paper:
        pid = p["pdf_id"]
        for pt in p.get("topics", []):
            if pt["id"] in member_map:
                edges.append({"data": {"source": pid, "target": member_map[pt["id"]], "rel": "member", "weight": 0.8}})
                connected_paper_ids.add(pid)

    # Handle orphan papers: connect to "Unrelated" topic
    all_paper_ids = {r["pdf_id"] for _, r in papers_df.iterrows()}
    orphan_paper_ids = all_paper_ids - connected_paper_ids
    
    unmatched_topics_from_orphans = []
    if orphan_paper_ids:
        for p in per_paper:
            if p["pdf_id"] in orphan_paper_ids:
                for pt in p.get("topics", []):
                    if pt["id"] not in member_map:
                        unmatched_topics_from_orphans.append(pt["label"])

        unrelated_topic_id = "T_Unrelated"
        unrelated_node_data = {
            "id": unrelated_topic_id,
            "label": "Unrelated",
            "type": "topic",
            "size": 40,
            "summary": "Papers with no clear topic match"
        }
        if unmatched_topics_from_orphans:
            unrelated_node_data["unmatched_topics"] = sorted(list(set(unmatched_topics_from_orphans)))
        
        nodes.append({"data": unrelated_node_data})

        for pid in orphan_paper_ids:
            edges.append({"data": {"source": pid, "target": unrelated_topic_id, "rel": "member", "weight": 0.8}})

    chunk_lookup = {}
    for paper_chunks in papers_content.values():
        for chunk in paper_chunks:
            if isinstance(chunk, dict):
                chunk_id = chunk.get("chunk_id")
                if chunk_id:
                    chunk_lookup[chunk_id] = chunk.get("text", "")

    # topic ↔ technique edges: link if variant shows up in any evidence text
    connected_technique_ids = set()
    tech_index = {g["canonical"]: set(g.get("variants", [])) | {g["canonical"]} for g in norm_tech.get("groups", [])}
    for t in corpus_topics.get("topics", []):
        tid = t["id"]
        # collect evidence snippets for member paper-topics
        member_ids = set(t.get("member_topic_ids", []))
        snippets = []
        for p in per_paper:
            for pt in p.get("topics", []):
                if pt["id"] in member_ids:
                    for ev in pt.get("evidence_chunks", []):
                        rationale = ev.get("rationale")
                        if rationale:
                            snippets.append(rationale)
                        chunk_text = chunk_lookup.get(ev.get("chunk_id"))
                        if chunk_text:
                            snippets.append(chunk_text)
        joined = " \n ".join(snippets).lower()
        for canon, variants in tech_index.items():
            if any(v.lower() in joined for v in variants if v):
                tech_id = f"Tech:{slug(canon)}"
                edges.append({"data": {"source": tid, "target": tech_id, "rel": "mentions", "weight": 1}})
                connected_technique_ids.add(tech_id)

    # Add only connected technique nodes to the graph
    nodes.extend([n for n in tech_nodes_to_add if n["data"]["id"] in connected_technique_ids])

    # topic ↔ topic edges: use embedding similarity of topic labels
    labels = [t["label"] for t in corpus_topics.get("topics", [])]
    tids = [t["id"] for t in corpus_topics.get("topics", [])]
    if labels:
        vecs = embed(labels)
        import numpy as np
        from numpy.linalg import norm
        import itertools
        def cos(a, b):
            import numpy as np
            return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-9))
        for i, j in itertools.combinations(range(len(vecs)), 2):
            s = cos(vecs[i], vecs[j])
            if s >= 0.3:
                edges.append({"data": {"source": tids[i], "target": tids[j], "rel": "similar", "weight": round(s, 3)}})

    # Fourth pass: find papers for orphan topics or prune them
    paper_title_map = {r["pdf_id"]: r["title"] for _, r in papers_df.iterrows()}
    tech_name_map = {f"Tech:{slug(g['canonical'])}": g["canonical"] for g in norm_tech.get("groups", [])}
    topic_context = {t["id"]: {"label": t["label"], "papers": set(), "techs": set(), "subtopics": set()} for t in corpus_topics.get("topics", [])}
    
    all_paper_topics_map = {topic["id"]: topic["label"] for paper in per_paper for topic in paper.get("topics", [])}
    for sub_id, main_id in subtopic_mapping.items():
        if main_id in topic_context:
            topic_context[main_id]["subtopics"].add(all_paper_topics_map.get(sub_id, sub_id))

    for edge in edges:
        src, tgt = edge["data"]["source"], edge["data"]["target"]
        rel = edge["data"].get("rel")
        if rel == "member" and tgt in topic_context:  # paper -> topic
            topic_context[tgt]["papers"].add(paper_title_map.get(src, src))
        elif rel == "mentions" and src in topic_context:  # topic -> tech
            topic_context[src]["techs"].add(tech_name_map.get(tgt, tgt))

    orphan_topic_ids = [tid for tid, context in topic_context.items() if not context["papers"]]
    if orphan_topic_ids:
        print(f"\nFound {len(orphan_topic_ids)} orphan topics. Attempting to find relevant papers...")
        topic_info_for_relinking = {t["id"]: {"label": t["label"], "summary": t.get("summary", "")} for t in corpus_topics.get("topics", [])}
        
        with ThreadPoolExecutor(max_workers=threads) as executor:
            future_to_topic = {
                executor.submit(
                    find_relevant_papers_for_topic,
                    topic_info_for_relinking[tid]["label"],
                    topic_info_for_relinking[tid]["summary"],
                    papers_content
                ): tid for tid in orphan_topic_ids
            }
            for future in as_completed(future_to_topic):
                topic_id = future_to_topic[future]
                try:
                    found_paper_ids = future.result()
                    if found_paper_ids:
                        print(f"  - Relinked topic '{topic_context[topic_id]['label']}' with {len(found_paper_ids)} paper(s).")
                        for paper_id in found_paper_ids:
                            edges.append({"data": {"source": paper_id, "target": topic_id, "rel": "member", "weight": 0.75}})
                            topic_context[topic_id]["papers"].add(paper_title_map.get(paper_id, paper_id))
                    else:
                        print(f"  - Pruning topic '{topic_context[topic_id]['label']}' (no relevant papers found).")
                        topic_context.pop(topic_id, None)
                except Exception as e:
                    print(f"  - Error processing topic {topic_context[topic_id]['label']}: {e}")

    # Filter out pruned topics
    all_corpus_topic_ids = {t["id"] for t in corpus_topics.get("topics", [])}
    final_topic_ids = set(topic_context.keys())
    pruned_topic_ids = all_corpus_topic_ids - final_topic_ids

    if pruned_topic_ids:
        nodes = [n for n in nodes if n['data']['id'] not in pruned_topic_ids]
        edges = [e for e in edges if e['data']['source'] not in pruned_topic_ids and e['data']['target'] not in pruned_topic_ids]
    
    # Third pass: summarize topics based on connections
    topic_summaries = {}
    with ThreadPoolExecutor(max_workers=threads) as executor:
        # Filter topic_context to only include non-pruned topics
        valid_topic_context = {k: v for k, v in topic_context.items() if k in final_topic_ids}
        futures = {
            executor.submit(summarize_topic, v["label"], sorted(list(v["papers"])), sorted(list(v["techs"]))): k
            for k, v in valid_topic_context.items()
        }
        for future in as_completed(futures):
            topic_id = futures[future]
            try:
                summary = future.result()
                topic_summaries[topic_id] = summary
            except Exception as e:
                print(f"Error summarizing topic {topic_id}: {e}")

    for n in nodes:
        if n["data"]["type"] == "topic":
            if n["data"]["id"] in topic_summaries:
                n["data"]["summary"] = topic_summaries[n["data"]["id"]]
            if n["data"]["id"] in topic_context and topic_context[n["data"]["id"]]["subtopics"]:
                n["data"]["subtopics"] = sorted(list(topic_context[n["data"]["id"]]["subtopics"]))

    # Extract key ideas for each topic
    topic_ideas = {}
    with ThreadPoolExecutor(max_workers=threads) as executor:
        future_to_topic = {
            executor.submit(
                extract_ideas_for_topic,
                topic_context[topic_id]["label"],
                topic_summaries.get(topic_id, "")
            ): topic_id
            for topic_id in final_topic_ids if topic_summaries.get(topic_id)
        }
        for future in as_completed(future_to_topic):
            topic_id = future_to_topic[future]
            try:
                ideas_data = future.result()
                topic_ideas[topic_id] = ideas_data.ideas
            except Exception as e:
                print(f"Error extracting ideas for topic {topic_context[topic_id]['label']}: {e}")

    # Enrich papers with summaries and topics
    with ThreadPoolExecutor(max_workers=threads) as executor:
        future_to_paper = {
            executor.submit(enrich_paper, papers_content[paper_id]): paper_id
            for paper_id in all_paper_ids
        }
        for future in as_completed(future_to_paper):
            paper_id = future_to_paper[future]
            try:
                enrichment_data = future.result()
                for n in nodes:
                    if n["data"]["id"] == paper_id:
                        n["data"]["summary"] = enrichment_data.summary
                        n["data"]["topics"] = enrichment_data.topics
                        break
            except Exception as e:
                print(f"Error enriching paper {paper_id}: {e}")

    # Fifth pass: enrich techniques with summaries and explanations
    connected_tech_ids = {n["data"]["id"] for n in nodes if n["data"]["type"] == "technique"}
    tech_context = {tech_id: {"name": tech_name_map.get(tech_id, tech_id), "topics": set()} for tech_id in connected_tech_ids}
    
    for edge in edges:
        src, tgt = edge["data"]["source"], edge["data"]["target"]
        rel = edge["data"].get("rel")
        if rel == "mentions" and src in final_topic_ids and tgt in tech_context: # topic -> tech
            tech_context[tgt]["topics"].add(topic_context[src]["label"])

    with ThreadPoolExecutor(max_workers=threads) as executor:
        future_to_tech = {
            executor.submit(enrich_technique, v["name"], sorted(list(v["topics"]))): k
            for k, v in tech_context.items() if v["topics"]
        }
        for future in as_completed(future_to_tech):
            tech_id = future_to_tech[future]
            try:
                enrichment_data = future.result()
                for n in nodes:
                    if n["data"]["id"] == tech_id:
                        n["data"]["summary"] = enrichment_data.summary
                        n["data"]["how_it_works"] = enrichment_data.how_it_works
                        n["data"]["relevance"] = enrichment_data.relevance
                        break
            except Exception as e:
                print(f"Error enriching technique {tech_context[tech_id]['name']}: {e}")

    # Generate markdown report
    generate_markdown_report(topic_context, topic_summaries, paper_title_map, out_dir, threads)

    # Export to Markdown Outline
    export_to_markdown_outline(topic_context, topic_ideas, out_dir)

    graph = {"elements": {"nodes": nodes, "edges": edges}}

    # write
    import os, json
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/graph.json", "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False)
    with open(f"{out_dir}/graph.html", "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE.replace("__GRAPH_JSON__", json.dumps(graph)))


def generate_markdown_report(topic_context: dict, topic_summaries: dict, paper_title_map: dict, out_dir: str, threads: int = 10):
    """Generates a markdown report with topic summaries, key insights, and paper lists."""
    from ..llm.client import chat

    _INSIGHTS_SYS = (
        "You are a research analyst. Given a topic summary, extract the 3 most important ideas or insights. "
        "Return only the 3 bullet points, one per line, starting with '- '. Be concise and specific."
    )

    # Sort topics by label for consistent ordering
    sorted_topics = sorted(topic_context.items(), key=lambda x: x[1]["label"])

    # Generate insights for all topics in parallel
    insights_map = {}
    with ThreadPoolExecutor(max_workers=threads) as executor:
        future_to_topic = {
            executor.submit(
                chat,
                _INSIGHTS_SYS,
                f"Topic: {context['label']}\n\nSummary: {topic_summaries.get(topic_id, 'No summary available.')}"
            ): topic_id
            for topic_id, context in sorted_topics
        }
        for future in as_completed(future_to_topic):
            topic_id = future_to_topic[future]
            try:
                insights_map[topic_id] = future.result()
            except Exception as e:
                print(f"Error generating insights for {topic_context[topic_id]['label']}: {e}")
                insights_map[topic_id] = None

    # Assemble markdown in original sorted order
    markdown_content = ["# Research Topics Report\n"]

    for topic_id, context in sorted_topics:
        topic_label = context["label"]
        summary = topic_summaries.get(topic_id, "No summary available.")
        papers = sorted(list(context["papers"]))
        techniques = sorted(list(context["techs"]))

        markdown_content.append(f"## {topic_label}\n")
        markdown_content.append(f"**Summary:**\n{summary}\n")

        insights = insights_map.get(topic_id)
        if insights:
            markdown_content.append("**Key Insights:**")
            markdown_content.append(insights)
            markdown_content.append("")
        else:
            markdown_content.append("**Key Insights:**\n- Unable to generate insights\n")

        if papers:
            markdown_content.append("**Relevant Papers:**")
            for paper in papers:
                markdown_content.append(f"- {paper}")
            markdown_content.append("")
        else:
            markdown_content.append("**Relevant Papers:** None\n")

        if techniques:
            markdown_content.append("**Related Techniques:**")
            for technique in techniques:
                markdown_content.append(f"- {technique}")
            markdown_content.append("")
        else:
            markdown_content.append("**Related Techniques:** None\n")

        markdown_content.append("---\n")

    # Write markdown file
    import os
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/topics_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_content))


def slug(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
