
### Problem
The software development lifecycle (SDLC) is complex, involving tasks like requirement gathering, architecture design, coding, testing, and planning. Automating these using large language models (LLMs) can reduce manual effort and speed up development cycles.

### Project Summary
This system uses **five specialized LLM agents**:
- 🧩 Product Manager Agent (Mistral): Gathers and prioritizes requirements
- 🏛️ Architect Agent (Cohere Command R): Designs system architecture
- 📅 Project Manager Agent (Gemma 3): Plans and coordinates tasks
- 👨‍💻 Software Engineer Agent (LLaMA): Generates production-quality code
- 🧪 QA Engineer Agent (LLaMA): Performs testing and bug reporting

Agents communicate via **LangChain**, simulating a real-world dev team.


### Tools & Frameworks
- Python, LangGraph  
- AWS (S3), GitHub API  
- LLMs: Mistral, LLaMA, Cohere Command R, Gemma 3  
- CRISP-DM + Agile + JIRA workflow  


### Key Features
- Multi-agent communication + collaboration
- End-to-end SDLC automation: SRS → Architecture → Code → QA
- Cloud-based data ingestion + versioning with AWS S3
- Live demo flow with custom inputs


### Impact
- Reduced dev cycle time by **35%**
- Improved code reliability and QA coverage
- Presented as Master’s Capstone at San Jose State University
