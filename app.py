#!/usr/bin/env python3
"""
PDFs to Mind Maps - Dash Web Application
Transform a folder of PDFs into an interactive mind map.
"""

import os
import time
import webbrowser
from pathlib import Path
import traceback
import tkinter as tk
from tkinter import filedialog
import threading

import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
from dotenv import load_dotenv

# Ensure environment variables (e.g., OPENAI_API_KEY) are loaded
load_dotenv()

# Import the pipeline entry point
from src.main import run as run_pipeline
import json
import threading
import queue


def path_exists_and_has_pdfs(pdf_dir: Path) -> bool:
    """Check if directory exists and contains PDF files"""
    if not pdf_dir.is_dir():
        return False
    return any(pdf_dir.rglob("*.pdf"))


def get_folder_info(folder_path: str) -> dict:
    """Get information about a folder"""
    try:
        path = Path(folder_path)
        if not path.exists():
            return {"exists": False, "error": "Path does not exist"}
        if not path.is_dir():
            return {"exists": False, "error": "Path is not a directory"}

        files = list(path.iterdir())
        file_count = len([f for f in files if f.is_file()])
        dir_count = len([f for f in files if f.is_dir()])
        pdf_count = len([f for f in files if f.is_file() and f.suffix.lower() == '.pdf'])

        return {
            "exists": True,
            "file_count": file_count,
            "dir_count": dir_count,
            "pdf_count": pdf_count,
            "is_pdf_folder": pdf_count > 0
        }
    except Exception as e:
        return {"exists": False, "error": str(e)}


def open_folder_dialog(initial_dir: str = None) -> str:
    """Open native OS folder dialog and return selected path"""
    try:
        # Create root window but don't show it
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        # Open folder dialog
        folder_path = filedialog.askdirectory(
            title="Select Folder",
            initialdir=initial_dir if initial_dir and Path(initial_dir).exists() else None
        )

        # Clean up
        root.destroy()

        return folder_path if folder_path else None

    except Exception as e:
        print(f"Error opening folder dialog: {e}")
        return None


# Global progress queue for tracking analysis progress
progress_queue = queue.Queue()

def progress_callback(step: str, progress: float, message: str = ""):
    """Callback function to update progress"""
    try:
        progress_queue.put({
            "step": step,
            "progress": progress,
            "message": message,
            "running": True
        }, block=False)
    except queue.Full:
        pass  # Skip if queue is full

def run_pipeline_with_progress(pdf_dir: Path, out_dir: Path, topics: int, chunk_words: int, chunk_overlap: int, threads: int):
    """Run the pipeline with progress tracking"""
    try:
        progress_callback("Initializing", 5, "Setting up analysis parameters...")

        # Import required modules
        from pathlib import Path
        from tqdm import tqdm
        import pandas as pd
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from src.pdf.extract import PDFExtractor
        from src.steps.section_labeler import label_section
        from src.steps.generic_section_labeler import label_generic_section
        from src.steps.per_paper_extract import extract_paper
        from src.steps.generic_document_extract import extract_generic_document
        from src.steps.content_type_detector import detect_content_type
        from src.steps.cluster_topics import cluster_topics
        from src.steps.normalize_techniques import normalize_techniques
        from src.graph.build import build_graph_and_write

        progress_callback("Scanning PDFs", 10, "Finding PDF files...")

        out_dir.mkdir(parents=True, exist_ok=True)
        pdfs = sorted([p for p in Path(pdf_dir).glob("**/*.pdf")])
        if not pdfs:
            raise SystemExit(f"No PDFs found in {pdf_dir}")

        progress_callback("Preparing", 15, f"Found {len(pdfs)} PDF files. Initializing extractor...")

        extractor = PDFExtractor(chunk_words=chunk_words, overlap_words=chunk_overlap)

        per_paper_results = []
        paper_rows = []
        paper_level_topics_flat = []
        technique_candidates = set()
        papers_content = {}

        progress_callback("Processing PDFs", 20, f"Starting parallel processing with {threads} threads...")

        # Define the single PDF processing function with progress
        def process_single_pdf_with_progress(args):
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

        # Prepare arguments for threading
        pdf_args = [(idx, pdf, extractor) for idx, pdf in enumerate(pdfs)]

        # Process PDFs in parallel with progress tracking
        completed_count = 0
        with ThreadPoolExecutor(max_workers=threads) as executor:
            # Submit all tasks
            future_to_pdf = {executor.submit(process_single_pdf_with_progress, args): args for args in pdf_args}

            # Collect results with progress updates
            for future in as_completed(future_to_pdf):
                try:
                    result = future.result()
                    completed_count += 1

                    if result is None:
                        continue  # Skip failed extractions

                    # Update progress (PDFs take 20-70% of total time)
                    pdf_progress = 20 + (completed_count / len(pdfs)) * 50
                    progress_callback("Processing PDFs", pdf_progress,
                                    f"Processed {completed_count}/{len(pdfs)} PDFs: {result['title']}")
                
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
                    completed_count += 1
                    pdf_progress = 20 + (completed_count / len(pdfs)) * 50
                    progress_callback("Processing PDFs", pdf_progress,
                                    f"Error processing {pdf_path.name}: {str(e)}")

        progress_callback("Clustering Topics", 75, "Analyzing and clustering corpus topics...")

        # cluster corpus topics
        corpus = cluster_topics(paper_level_topics_flat, k=topics)
        corpus_dict = json.loads(corpus.model_dump_json())

        progress_callback("Normalizing Techniques", 85, "Normalizing technique names...")

        # normalize techniques
        norm = normalize_techniques(sorted(list(technique_candidates)))
        norm_dict = json.loads(norm.model_dump_json())

        progress_callback("Building Graph", 95, "Building interactive mind map...")

        # write graph & viewer
        papers_df = pd.DataFrame(paper_rows)
        build_graph_and_write(papers_df, per_paper_results, corpus_dict, norm_dict, str(out_dir), papers_content)

        progress_callback("Complete", 100, "Analysis completed successfully!")

        # Signal completion with output directory info
        progress_queue.put({
            "running": False,
            "completed": True,
            "output_dir": str(out_dir)
        })

    except Exception as e:
        progress_queue.put({
            "running": False,
            "error": True,
            "message": f"Analysis failed: {str(e)}"
        })
        raise


# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "PDFs → Mind Maps"

# Resolve project defaults
project_root = Path(__file__).parent.resolve()
default_papers = str((project_root / "papers").resolve())
default_output = str((project_root / "output").resolve())

# Common folder suggestions
common_folders = []
if os.name == "nt":  # Windows
    common_folders = [
        ("C:\\", "C: Drive"),
        ("D:\\", "D: Drive"),
        (str(Path.home() / "Documents"), "Documents"),
        (str(Path.home() / "Downloads"), "Downloads"),
        (str(Path.home() / "Desktop"), "Desktop"),
    ]
else:  # Unix-like
    common_folders = [
        ("/", "Root"),
        (str(Path.home()), "Home"),
        (str(Path.home() / "Documents"), "Documents"),
        (str(Path.home() / "Downloads"), "Downloads"),
    ]

# Filter to existing paths
common_folders = [(path, name) for path, name in common_folders if Path(path).exists()]

# Define the app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🧭 PDFs → Mind Maps", className="text-center mb-4"),
            html.P("Transform a folder of PDFs into an interactive mind map.",
                   className="text-center text-muted mb-5"),
        ])
    ]),

    dbc.Row([
        # Left column - Settings
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("⚙️ Settings")),
                dbc.CardBody([
                    # PDF Folder Section
                    html.H5("📁 PDFs Folder"),
                    html.P("Select the folder containing your PDF files", className="text-muted small"),

                    dbc.InputGroup([
                        dbc.Input(
                            id="pdf-folder-input",
                            placeholder="Enter folder path...",
                            value=default_papers,
                            type="text"
                        ),
                        dbc.Button("📂 Browse", id="pdf-browse-btn", color="secondary", outline=True)
                    ], className="mb-2"),

                    # Quick folder buttons for PDF folder
                    html.Div([
                        dbc.ButtonGroup([
                            dbc.Button("🏠 Home", id="pdf-home-btn", size="sm", color="light"),
                            dbc.Button("📁 Default", id="pdf-default-btn", size="sm", color="light"),
                            dbc.Button("⬆️ Parent", id="pdf-parent-btn", size="sm", color="light"),
                        ], className="mb-2 w-100")
                    ]),

                    # Common locations dropdown for PDF folder
                    dbc.Select(
                        id="pdf-common-select",
                        options=[{"label": f"📁 {name}", "value": path} for path, name in common_folders],
                        placeholder="Quick locations...",
                        className="mb-3"
                    ),

                    # PDF folder status
                    html.Div(id="pdf-folder-status", className="mb-4"),

                    html.Hr(),

                    # Output Folder Section
                    html.H5("📤 Output Folder"),
                    html.P("Select where to save the generated mind map and reports", className="text-muted small"),

                    dbc.InputGroup([
                        dbc.Input(
                            id="output-folder-input",
                            placeholder="Enter folder path...",
                            value=default_output,
                            type="text"
                        ),
                        dbc.Button("📂 Browse", id="output-browse-btn", color="secondary", outline=True)
                    ], className="mb-2"),

                    # Quick folder buttons for output folder
                    html.Div([
                        dbc.ButtonGroup([
                            dbc.Button("🏠 Home", id="output-home-btn", size="sm", color="light"),
                            dbc.Button("📁 Default", id="output-default-btn", size="sm", color="light"),
                            dbc.Button("⬆️ Parent", id="output-parent-btn", size="sm", color="light"),
                        ], className="mb-2 w-100")
                    ]),

                    # Common locations dropdown for output folder
                    dbc.Select(
                        id="output-common-select",
                        options=[{"label": f"📁 {name}", "value": path} for path, name in common_folders],
                        placeholder="Quick locations...",
                        className="mb-3"
                    ),

                    # Output folder status
                    html.Div(id="output-folder-status", className="mb-4"),

                    html.Hr(),

                    # Analysis Parameters
                    html.H5("🔧 Analysis Parameters"),

                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Number of topics"),
                            dbc.Input(
                                id="topics-input",
                                type="number",
                                min=1, max=50, step=1, value=12,
                                className="mb-3"
                            ),
                        ], md=6),
                        dbc.Col([
                            dbc.Label("Threads"),
                            dbc.Input(
                                id="threads-input",
                                type="number",
                                min=1, max=64, step=1, value=10,
                                className="mb-3"
                            ),
                        ], md=6),
                    ]),

                    # Advanced settings in collapsible section
                    dbc.Collapse([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Chunk words"),
                                dbc.Input(
                                    id="chunk-words-input",
                                    type="number",
                                    min=100, max=3000, step=50, value=900,
                                    className="mb-3"
                                ),
                            ], md=6),
                            dbc.Col([
                                dbc.Label("Chunk overlap"),
                                dbc.Input(
                                    id="chunk-overlap-input",
                                    type="number",
                                    min=0, max=1000, step=10, value=120,
                                    className="mb-3"
                                ),
                            ], md=6),
                        ]),
                    ], id="advanced-collapse"),

                    dbc.Button(
                        "⚙️ Advanced Settings",
                        id="advanced-toggle-btn",
                        color="link",
                        size="sm",
                        className="mb-3"
                    ),
                ])
            ])
        ], md=4),

        # Right column - Main content
        dbc.Col([
            # Validation alerts
            html.Div(id="validation-alerts", className="mb-3"),

            # Run button
            dbc.Button(
                "🚀 Run Analysis",
                id="run-btn",
                color="primary",
                size="lg",
                disabled=True,
                className="w-100 mb-4"
            ),

            # Progress and results area
            html.Div(id="progress-area"),
            html.Div(id="results-area"),

        ], md=8),
    ]),

    # Store components for state management
    dcc.Store(id="pdf-folder-store", data=default_papers),
    dcc.Store(id="output-folder-store", data=default_output),
    dcc.Store(id="pdf-browse-store", data=0),
    dcc.Store(id="output-browse-store", data=0),
    dcc.Store(id="progress-store", data={"running": False, "step": "", "progress": 0, "message": ""}),
    dcc.Interval(id="progress-interval", interval=500, n_intervals=0, disabled=True),

], fluid=True, className="py-4")


# Callback for advanced settings toggle
@app.callback(
    Output("advanced-collapse", "is_open"),
    Input("advanced-toggle-btn", "n_clicks"),
    State("advanced-collapse", "is_open"),
)
def toggle_advanced(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


# Callback for PDF folder management
@app.callback(
    [Output("pdf-folder-store", "data"),
     Output("pdf-folder-input", "value"),
     Output("pdf-browse-store", "data")],
    [Input("pdf-folder-input", "value"),
     Input("pdf-home-btn", "n_clicks"),
     Input("pdf-default-btn", "n_clicks"),
     Input("pdf-parent-btn", "n_clicks"),
     Input("pdf-common-select", "value"),
     Input("pdf-browse-btn", "n_clicks")],
    [State("pdf-folder-store", "data"),
     State("pdf-browse-store", "data")],
    prevent_initial_call=True
)
def update_pdf_folder(input_value, home_clicks, default_clicks, parent_clicks,
                      common_value, browse_clicks, current_folder, browse_counter):

    if ctx.triggered_id == "pdf-folder-input":
        return input_value, input_value, browse_counter
    elif ctx.triggered_id == "pdf-home-btn":
        home_path = str(Path.home())
        return home_path, home_path, browse_counter
    elif ctx.triggered_id == "pdf-default-btn":
        return default_papers, default_papers, browse_counter
    elif ctx.triggered_id == "pdf-parent-btn":
        try:
            parent_path = str(Path(current_folder).parent)
            return parent_path, parent_path, browse_counter
        except:
            return current_folder, current_folder, browse_counter
    elif ctx.triggered_id == "pdf-common-select" and common_value:
        return common_value, common_value, browse_counter
    elif ctx.triggered_id == "pdf-browse-btn":
        # Open native folder dialog
        selected_folder = open_folder_dialog(current_folder)
        if selected_folder:
            return selected_folder, selected_folder, browse_counter + 1
        else:
            return current_folder, current_folder, browse_counter + 1

    return current_folder, current_folder, browse_counter


# Callback for output folder management
@app.callback(
    [Output("output-folder-store", "data"),
     Output("output-folder-input", "value"),
     Output("output-browse-store", "data")],
    [Input("output-folder-input", "value"),
     Input("output-home-btn", "n_clicks"),
     Input("output-default-btn", "n_clicks"),
     Input("output-parent-btn", "n_clicks"),
     Input("output-common-select", "value"),
     Input("output-browse-btn", "n_clicks")],
    [State("output-folder-store", "data"),
     State("output-browse-store", "data")],
    prevent_initial_call=True
)
def update_output_folder(input_value, home_clicks, default_clicks, parent_clicks,
                        common_value, browse_clicks, current_folder, browse_counter):

    if ctx.triggered_id == "output-folder-input":
        return input_value, input_value, browse_counter
    elif ctx.triggered_id == "output-home-btn":
        home_path = str(Path.home())
        return home_path, home_path, browse_counter
    elif ctx.triggered_id == "output-default-btn":
        return default_output, default_output, browse_counter
    elif ctx.triggered_id == "output-parent-btn":
        try:
            parent_path = str(Path(current_folder).parent)
            return parent_path, parent_path, browse_counter
        except:
            return current_folder, current_folder, browse_counter
    elif ctx.triggered_id == "output-common-select" and common_value:
        return common_value, common_value, browse_counter
    elif ctx.triggered_id == "output-browse-btn":
        # Open native folder dialog
        selected_folder = open_folder_dialog(current_folder)
        if selected_folder:
            return selected_folder, selected_folder, browse_counter + 1
        else:
            return current_folder, current_folder, browse_counter + 1

    return current_folder, current_folder, browse_counter


# Callback for folder status updates
@app.callback(
    [Output("pdf-folder-status", "children"),
     Output("output-folder-status", "children")],
    [Input("pdf-folder-store", "data"),
     Input("output-folder-store", "data")]
)
def update_folder_status(pdf_folder, output_folder):
    # PDF folder status
    pdf_info = get_folder_info(pdf_folder)
    if not pdf_info["exists"]:
        pdf_status = dbc.Alert(f"❌ {pdf_info.get('error', 'Invalid path')}", color="danger", className="mb-0")
    elif pdf_info["pdf_count"] > 0:
        pdf_status = dbc.Alert(f"✅ {pdf_info['pdf_count']} PDF files found", color="success", className="mb-0")
    else:
        pdf_status = dbc.Alert("⚠️ No PDF files found", color="warning", className="mb-0")

    # Output folder status
    output_info = get_folder_info(output_folder)
    if not output_info["exists"]:
        # Check if parent directory exists for output folder
        try:
            parent = Path(output_folder).parent
            if parent.exists():
                output_status = dbc.Alert("✅ Will create output folder", color="info", className="mb-0")
            else:
                output_status = dbc.Alert(f"❌ {output_info.get('error', 'Invalid path')}", color="danger", className="mb-0")
        except:
            output_status = dbc.Alert("❌ Invalid output path", color="danger", className="mb-0")
    else:
        output_status = dbc.Alert("✅ Output folder exists", color="success", className="mb-0")

    return pdf_status, output_status


# Callback for validation and run button
@app.callback(
    [Output("validation-alerts", "children"),
     Output("run-btn", "disabled")],
    [Input("pdf-folder-store", "data"),
     Input("output-folder-store", "data")]
)
def update_validation(pdf_folder, output_folder):
    alerts = []
    can_run = True

    # Validate PDF folder
    pdf_info = get_folder_info(pdf_folder)
    if not pdf_info["exists"]:
        alerts.append(dbc.Alert("❌ PDF folder does not exist", color="danger"))
        can_run = False
    elif pdf_info["pdf_count"] == 0:
        alerts.append(dbc.Alert("⚠️ No PDF files found in selected folder", color="warning"))
        can_run = False

    # Validate output folder
    output_info = get_folder_info(output_folder)
    if not output_info["exists"]:
        try:
            parent = Path(output_folder).parent
            if not parent.exists():
                alerts.append(dbc.Alert("❌ Output folder parent directory does not exist", color="danger"))
                can_run = False
        except:
            alerts.append(dbc.Alert("❌ Invalid output folder path", color="danger"))
            can_run = False

    if can_run and not alerts:
        alerts.append(dbc.Alert("✅ Ready to run analysis", color="success"))

    return alerts, not can_run


# Add progress update callback
@app.callback(
    [Output("progress-store", "data"),
     Output("progress-interval", "disabled")],
    Input("progress-interval", "n_intervals"),
    State("progress-store", "data"),
    prevent_initial_call=True
)
def update_progress(n_intervals, current_progress):
    """Update progress from the global queue"""
    try:
        # Check for new progress updates
        while not progress_queue.empty():
            try:
                new_progress = progress_queue.get_nowait()
                current_progress.update(new_progress)
                if not new_progress.get("running", True):
                    # Analysis finished or errored, disable interval
                    return current_progress, True
            except queue.Empty:
                break

        return current_progress, current_progress.get("running", False)
    except:
        return current_progress, True


# Callback for running the analysis
@app.callback(
    [Output("progress-area", "children"),
     Output("results-area", "children"),
     Output("run-btn", "disabled", allow_duplicate=True),
     Output("progress-store", "data", allow_duplicate=True),
     Output("progress-interval", "disabled", allow_duplicate=True)],
    Input("run-btn", "n_clicks"),
    [State("pdf-folder-store", "data"),
     State("output-folder-store", "data"),
     State("topics-input", "value"),
     State("chunk-words-input", "value"),
     State("chunk-overlap-input", "value"),
     State("threads-input", "value")],
    prevent_initial_call=True
)
def run_analysis(n_clicks, pdf_folder, output_folder, topics, chunk_words, chunk_overlap, threads):
    if not n_clicks:
        return "", "", False, {"running": False, "step": "", "progress": 0, "message": ""}, True

    # Clear the progress queue
    while not progress_queue.empty():
        try:
            progress_queue.get_nowait()
        except queue.Empty:
            break

    # Start the analysis in a separate thread
    def run_analysis_thread():
        try:
            run_pipeline_with_progress(
                pdf_dir=Path(pdf_folder),
                out_dir=Path(output_folder),
                topics=int(topics),
                chunk_words=int(chunk_words),
                chunk_overlap=int(chunk_overlap),
                threads=int(threads)
            )
        except Exception as e:
            # Error handling is done in the progress callback
            pass

    # Start the analysis thread
    analysis_thread = threading.Thread(target=run_analysis_thread, daemon=True)
    analysis_thread.start()

    # Return initial progress display
    initial_progress = dbc.Card([
        dbc.CardBody([
            html.H5("🚀 Starting Analysis...", className="card-title"),
            html.P("Initializing the mind map generation process..."),
            dbc.Progress(value=0, striped=True, animated=True, color="primary", className="mb-3"),
            html.P(f"📁 PDFs: {pdf_folder}", className="small text-muted"),
            html.P(f"📤 Output: {output_folder}", className="small text-muted"),
            html.P(f"⚙️ Config: {topics} topics, {threads} threads, {chunk_words}/{chunk_overlap} chunk", className="small text-muted"),
        ])
    ], color="primary", outline=True)

    # Initialize progress store and enable interval
    initial_progress_data = {"running": True, "step": "Starting", "progress": 0, "message": "Initializing analysis..."}

    return initial_progress, "", True, initial_progress_data, False


# Callback to update progress display
@app.callback(
    [Output("progress-area", "children", allow_duplicate=True),
     Output("results-area", "children", allow_duplicate=True),
     Output("run-btn", "disabled", allow_duplicate=True),
     Output("progress-interval", "disabled", allow_duplicate=True)],
    Input("progress-store", "data"),
    prevent_initial_call=True
)
def display_progress(progress_data):
    if not progress_data.get("running", False):
        # Analysis completed or errored
        if progress_data.get("completed", False):
            # Success
            results_content = []

            # Success message
            results_content.append(
                dbc.Alert("🎉 Analysis completed successfully!", color="success", className="mb-3")
            )

            # Get output directory and create file links
            output_dir = progress_data.get("output_dir", "output")
            output_path = Path(output_dir)

            # Define the files to check for
            output_files = [
                {
                    "name": "Interactive Mind Map",
                    "file": "graph.html",
                    "icon": "📊",
                    "description": "Interactive HTML visualization of the mind map",
                    "primary": True
                },
                {
                    "name": "Topics Report",
                    "file": "topics_report.md",
                    "icon": "📄",
                    "description": "Detailed analysis of extracted topics"
                },
                {
                    "name": "Mind Map Outline",
                    "file": "mindmap_outline.md",
                    "icon": "📋",
                    "description": "Structured outline of the mind map"
                },
                {
                    "name": "Graph Data",
                    "file": "graph.json",
                    "icon": "📈",
                    "description": "Raw graph data in JSON format"
                }
            ]

            # Create file cards with links
            results_content.append(html.H5("📁 Generated Files:", className="mb-3"))

            for file_info in output_files:
                file_path = output_path / file_info["file"]
                if file_path.exists():
                    # Create file URI for local access
                    file_uri = file_path.as_uri()

                    # Auto-open the main graph file
                    if file_info.get("primary", False):
                        try:
                            webbrowser.open_new_tab(file_uri)
                            open_msg = " (opened in browser)"
                        except:
                            open_msg = ""
                    else:
                        open_msg = ""

                    # Create card with clickable link
                    card_content = [
                        dbc.CardBody([
                            html.H6([
                                file_info["icon"], " ", file_info["name"], open_msg
                            ], className="card-title"),
                            html.P(file_info["description"], className="card-text small mb-2"),
                            html.A(
                                f"📂 Open {file_info['file']}",
                                href=file_uri,
                                target="_blank",
                                className="btn btn-outline-primary btn-sm"
                            ),
                            html.P(
                                f"Location: {file_path}",
                                className="small text-muted mt-2 font-monospace"
                            ),
                        ])
                    ]

                    card_color = "success" if file_info.get("primary", False) else "light"
                    results_content.append(
                        dbc.Card(card_content, color=card_color, outline=True, className="mb-2")
                    )

            # Add instructions if files not found
            if not any((output_path / f["file"]).exists() for f in output_files):
                results_content.append(
                    dbc.Alert(
                        f"⚠️ Output files not found in {output_dir}. Check the console for errors.",
                        color="warning"
                    )
                )

            return "", results_content, False, True

        elif progress_data.get("error", False):
            # Error
            error_card = dbc.Card([
                dbc.CardBody([
                    html.H5("❌ Analysis Failed", className="card-title text-danger"),
                    html.P(progress_data.get("message", "Unknown error occurred"), className="card-text"),
                ])
            ], color="danger", outline=True)

            return "", error_card, False, True

    # Analysis running - show progress
    step = progress_data.get("step", "Initializing")
    progress = progress_data.get("progress", 0)
    message = progress_data.get("message", "")

    progress_card = dbc.Card([
        dbc.CardBody([
            html.H5(f"🔄 {step}", className="card-title"),
            html.P(message if message else "Processing..."),
            dbc.Progress(
                value=progress,
                striped=True,
                animated=True,
                color="primary",
                className="mb-2"
            ),
            html.P(f"{progress:.1f}% complete", className="small text-muted"),
        ])
    ], color="primary", outline=True)

    return progress_card, "", True, False


if __name__ == "__main__":
    print("Starting PDFs to Mind Maps application...")
    print("Browse to http://localhost:8050 to access the application")
    app.run(debug=True, port=8050)