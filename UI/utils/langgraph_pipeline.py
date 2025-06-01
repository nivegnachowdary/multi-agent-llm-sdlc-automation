import uuid, zipfile, re
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Tuple

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages.base import BaseMessage

from agents import (
    product_manager_agent,
    project_manager_agent,
    software_architect_agent,
    software_engineer_agent,
    quality_assurance_agent,
)

# ——————————————
# 1) State definitions
# ——————————————
class InputState(TypedDict):
    messages: List[BaseMessage]
    chat_log: List[Dict[str, Any]]

class OutputState(TypedDict):
    pm_output: str
    proj_output: str
    arch_output: str
    dev_output: str
    qa_output: str
    chat_log: List[Dict[str, Any]]

# ——————————————
# 2) Wrap agents so they see full history
# ——————————————
def wrap_agent(agent_run, output_key: str):
    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        history = state["messages"]
        log     = state["chat_log"]
        result  = agent_run({"messages": history, "chat_log": log})
        return {
            "messages": history + result["messages"],
            "chat_log":  result["chat_log"],
            output_key:  result[output_key],
        }
    return node

# ——————————————
# 3) Bridge → ProductManager
# ——————————————
def bridge_to_pm(state: Dict[str, Any]) -> Dict[str, Any]:
    history = state["messages"]
    log     = state["chat_log"]
    if not history or not isinstance(history[-1], HumanMessage):
        raise ValueError("bridge_to_pm expected a HumanMessage at history end")
    prompt = history[-1].content
    spec_prompt = (
        f"# Stakeholder Prompt\n\n"
        f"\"{prompt}\"\n\n"
        "Generate a structured product specification including:\n"
        "- Goals\n"
        "- Key features\n"
        "- User stories\n"
        "- Success metrics\n"
    )
    return {
        "messages": [AIMessage(content=spec_prompt)],
        "chat_log": log + [{"role": "System", "content": spec_prompt}],
    }

# ——————————————
# 4) Build & compile the LangGraph
# ——————————————
graph = StateGraph(input=InputState, output=OutputState)

graph.add_node("BridgePM",         bridge_to_pm)
graph.add_node("ProductManager",   wrap_agent(product_manager_agent.run,   "pm_output"))
graph.add_node("ProjectManager",   wrap_agent(project_manager_agent.run,   "proj_output"))
graph.add_node("SoftwareArchitect",wrap_agent(software_architect_agent.run, "arch_output"))
graph.add_node("SoftwareEngineer", wrap_agent(software_engineer_agent.run,  "dev_output"))
graph.add_node("QualityAssurance", wrap_agent(quality_assurance_agent.run,  "qa_output"))

graph.set_entry_point("BridgePM")
graph.add_edge("BridgePM",         "ProductManager")
graph.add_edge("ProductManager",   "ProjectManager")
graph.add_edge("ProjectManager",   "SoftwareArchitect")
graph.add_edge("SoftwareArchitect","SoftwareEngineer")
graph.add_edge("SoftwareEngineer", "QualityAssurance")
graph.add_edge("QualityAssurance", END)

compiled_graph = graph.compile()

# ——————————————
# 5) Parse spec into sections
# ——————————————
def parse_spec(spec: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    for m in re.finditer(r"##\s*(.+?)\n((?:- .+\n?)+)", spec):
        name = m.group(1).strip()
        items = [line[2:].strip() for line in m.group(2).splitlines() if line.startswith("- ")]
        sections[name] = items
    return sections

# ——————————————
# 6) Run pipeline, generate site, zip, return (chat_log, zip_path)
# ——————————————
def run_pipeline_and_save(prompt: str) -> Tuple[List[Dict[str, Any]], str]:
    # a) invoke agents
    initial_state = {"messages": [HumanMessage(content=prompt)], "chat_log": []}
    final_state   = compiled_graph.invoke(initial_state)

    chat_log  = final_state["chat_log"]
    qa_output = final_state["qa_output"]

    # b) parse spec
    spec = parse_spec(qa_output)
    features     = spec.get("Key features", [])
    testimonials = spec.get("User stories", [])

    # c) build HTML
    title = prompt.title()
    domain = prompt.replace(" ", "").lower() + ".com"
    cards_html = "\n".join(f"<div class='card'><h3>{f}</h3></div>" for f in features)
    test_html  = "\n".join(f"<blockquote>{t}</blockquote>" for t in testimonials)

    html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{title}</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header><h1>{title}</h1></header>
  <section id="features">
    <h2>Features</h2>
    <div class="cards">
      {cards_html}
    </div>
  </section>
  <section id="testimonials">
    <h2>Testimonials</h2>
    {test_html or '<p>No testimonials provided.</p>'}
  </section>
  <section id="contact">
    <h2>Contact Us</h2>
    <p>Email: info@{domain}</p>
  </section>
</body>
</html>"""

    # d) basic CSS
    css_code = """
body { font-family: Arial, sans-serif; margin: 1em; line-height: 1.5; }
header { text-align: center; margin-bottom: 2em; }
.cards { display: grid; grid-template-columns: repeat(auto-fit,minmax(150px,1fr)); gap: 1em; }
.card { background: #f9f9f9; padding: 1em; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; }
blockquote { font-style: italic; margin: 1em; padding: 0.5em; background: #eef; border-left: 4px solid #99f; }
"""

    # e) write & zip
    site_id  = uuid.uuid4().hex
    out_dir  = Path("output")
    site_dir = out_dir / f"site_{site_id}"
    site_dir.mkdir(parents=True, exist_ok=True)

    (site_dir / "index.html").write_text(html_code, encoding="utf-8")
    (site_dir / "styles.css").write_text(css_code,  encoding="utf-8")

    zip_path = out_dir / f"site_{site_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in site_dir.iterdir():
            zf.write(f, arcname=f.name)

    return chat_log, str(zip_path)
