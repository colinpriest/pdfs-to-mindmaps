from typing import Dict, Any, List
import os

def export_to_markdown_outline(topic_context: Dict[str, Any], topic_ideas: Dict[str, List[str]], out_dir: str):
    """
    Exports the topic and ideas data to a structured markdown file.
    """
    markdown_lines = ["# Research Topics\n"]
    
    # Sort topics by label for consistent ordering
    sorted_topics = sorted(topic_context.items(), key=lambda x: x[1]["label"])

    for topic_id, context in sorted_topics:
        topic_label = context["label"]
        ideas = topic_ideas.get(topic_id, [])
        
        # Add the main topic
        markdown_lines.append(f"- **{topic_label}**")
        
        # Add ideas as sub-bullets
        if ideas:
            for idea in ideas:
                markdown_lines.append(f"  - {idea}")
        
        markdown_lines.append("") # Add a blank line for readability

    # Write the markdown file
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "mindmap_outline.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))
    
    print(f"Successfully exported mind map outline to {md_path}")
