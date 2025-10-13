# PDFs to Mind Maps

Transform academic papers into interactive mind maps using AI-powered analysis.

## Overview

This tool processes a collection of PDF papers and creates comprehensive mind maps that visualize the relationships between topics, techniques, and papers. It uses ChatGPT (via Instructor + Pydantic) to extract structured insights from academic literature and presents them in multiple formats.

## What it does

1. **Extracts text** from PDFs using PyMuPDF
2. **Detects content type** using ChatGPT to classify documents as scientific papers or other types
3. **Analyzes content** using ChatGPT with appropriate prompts:
   - **Scientific papers**: Traditional academic analysis with sections like Abstract, Methods, Results
   - **Other documents**: Generic analysis suitable for business reports, articles, manuals, etc.
   - Extract topics and techniques/concepts with supporting evidence
   - Cluster related topics across the corpus
   - Normalize technique/concept names and variants
4. **Generates outputs**:
   - Interactive HTML graph (`graph.html`) with Cytoscape.js
   - Detailed topics report (`topics_report.md`) with summaries and insights
   - Mind map outline (`mindmap_outline.md`) for import into mind mapping tools
   - Raw graph data (`graph.json`) for further processing

## Key Features

- **Intelligent content detection** automatically classifies documents as scientific papers or other types
- **Adaptive processing** uses different analysis approaches based on document type
- **Multi-threaded processing** for faster PDF analysis
- **Structured extraction** using Pydantic models for reliable data parsing
- **Interactive visualization** with zoom, pan, and search capabilities
- **Multiple export formats** for different use cases
- **Configurable parameters** for chunking, topic clustering, and processing

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

```bash
# Clone and setup
git clone <repository-url>
cd pdfs-to-mindmaps

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
echo "OPENAI_API_KEY=your_api_key_here" > .env
# Optional: Set custom models
echo "CHAT_MODEL=gpt-4o-mini" >> .env
echo "EMBED_MODEL=text-embedding-3-large" >> .env
```

### Usage

```bash
# Place your PDF files in the ./papers/ directory
# Run the analysis
python -m src.main --pdf_dir ./papers --out_dir ./output --topics 12 --threads 10

# View results
# - Interactive graph: open output/graph.html
# - Topics report: open output/topics_report.md  
# - Mind map outline: import output/mindmap_outline.md into XMind or similar
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pdf_dir` | Required | Directory containing PDF files |
| `--out_dir` | `./output` | Output directory for results |
| `--topics` | `12` | Number of topic clusters (8-20 recommended) |
| `--chunk_words` | `900` | Words per text chunk |
| `--chunk_overlap` | `120` | Overlap between chunks |
| `--threads` | `10` | Concurrent PDF processing threads |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | Required | Your OpenAI API key |
| `OPENAI_BASE_URL` | None | Custom OpenAI API endpoint |
| `CHAT_MODEL` | `gpt-4o-mini` | Model for text analysis |
| `EMBED_MODEL` | `text-embedding-3-large` | Model for embeddings |

## Output Files

- **`graph.html`**: Interactive web-based mind map with search and filtering
- **`graph.json`**: Raw graph data in JSON format
- **`topics_report.md`**: Detailed analysis with topic summaries and insights
- **`mindmap_outline.md`**: Hierarchical outline for mind mapping tools

## Architecture

The pipeline consists of several processing steps:

1. **PDF Extraction** (`src/pdf/extract.py`): Text extraction and chunking
2. **Content Type Detection** (`src/steps/content_type_detector.py`): Classifies documents as scientific papers or other types
3. **Section Labeling**: Document structure analysis
   - **Scientific papers** (`src/steps/section_labeler.py`): Academic sections (Abstract, Methods, Results, etc.)
   - **Other documents** (`src/steps/generic_section_labeler.py`): Generic sections (Introduction, Main Content, etc.)
4. **Document Analysis**: Topic and technique/concept extraction
   - **Scientific papers** (`src/steps/per_paper_extract.py`): Research topics and techniques
   - **Other documents** (`src/steps/generic_document_extract.py`): General topics and concepts
5. **Topic Clustering** (`src/steps/cluster_topics.py`): Cross-document topic grouping
6. **Technique Normalization** (`src/steps/normalize_techniques.py`): Technique/concept name standardization
7. **Graph Building** (`src/graph/build.py`): Visualization generation

## Dependencies

- **Core**: OpenAI, Instructor, Pydantic, python-dotenv
- **PDF Processing**: PyMuPDF, BeautifulSoup4
- **Data Processing**: pandas, numpy, scikit-learn
- **Utilities**: tqdm for progress tracking

## License

[Add your license information here]`