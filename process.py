from llama_index.core import (
    VectorStoreIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core import SummaryIndex
from llama_index.core.schema import IndexNode
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.core.callbacks import CallbackManager
import os
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.agent import ReActAgent
import os
from autogen.agentchat.contrib.llamaindex_conversable_agent import LLamaIndexConversableAgent
# legal_group_chat.py
from typing import List, Dict, Any
import autogen
from autogen import Agent, AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI


chapter_dict = {'INTRODUCTION' : (21,28),
               'JURISDICTION': (29, 136),
               'THE EMPLOYMENT RELATIONSHIP': (137, 206),
               'INJURY':(207, 264),
               'COURSE OF EMPLOYMENT (TIME AND PLACE)':(265,302),
               'ARISING OUT OF EMPLOYMENT (CAUSAL RELATIONSHIP)': (303, 358),
               'TEMPORARY DISABILITY': (359, 406),
               'PERMANENT DISABILITY': (407, 498),
               'MEDICAL BENEFITS' : (499, 558),
               'LIENS AND MEDICAL-LEGAL COST PETITIONS': (559,608),
               'DEATH BENEFITS': (609, 638),
               'PENALTIES: Increased and Reduced Compensation on Account of Fault' : (639,731),
               'INSURANCE: Securing Liability for Compensation': (737,810),
               'THE DELIVERY SYSTEM: ADMINISTRATIVE SUPERVISION; NOTICE REQUIREMENTS; AUDIT PROCEDURES; ANTI-FRAUD LEGISLATION; INJURY PREVENTION PROGRAMS; INFORMATION AND ASSISTANCE PROGRAM; NEW SB 863 ORGANIZATIONS': (811,868),
               'SETTLEMENTS: Compromise and Release, and Stipulated Findings & Award': (869,934),
               'CLAIM FILING PROCEDURE, PLEADINGS, VENUE AND DISMISSAL': (935, 962),
               'ATTORNEYS AND REPRESENTATIVES': (963, 1012),
               'STATUTE OF LIMITATIONS: Original Filings and Reopenings': (1013, 1078),
               'PREPARATION FOR TRIAL AND OTHER PRE-TRIAL MATTERS': (1079, 1148),
               'PRE-TRIAL DISCOVERY: Depositions and Other Techniques' : (1149, 1164),
               'TRIAL SETTING AND TRIAL' : (1165, 1282),
               'JUDGMENTS: Post-Trial Dispositions, Decisions and Other Matters (Interest, Costs, Credit, Clerical Error, Enforcement, Commutation and Restitution)': (1283, 1326),
               'RECONSIDERATION AND REVIEW': (1327, 1388),
               'THIRD-PARTY SUITS': (1389, 1432)
               }

chapter_filenames = {
    'INTRODUCTION': 'introduction',
    'JURISDICTION': 'jurisdiction',
    'THE EMPLOYMENT RELATIONSHIP': 'employment',
    'INJURY': 'injury',
    'COURSE OF EMPLOYMENT (TIME AND PLACE)': 'employment_course',
    'ARISING OUT OF EMPLOYMENT (CAUSAL RELATIONSHIP)': 'employment_causal',
    'TEMPORARY DISABILITY': 'temp_disability',
    'PERMANENT DISABILITY': 'perm_disability',
    'MEDICAL BENEFITS': 'medical_benefits',
    'LIENS AND MEDICAL-LEGAL COST PETITIONS': 'liens_petitions',
    'DEATH BENEFITS': 'death_benefits',
    'PENALTIES: Increased and Reduced Compensation on Account of Fault': 'penalties',
    'INSURANCE: Securing Liability for Compensation': 'insurance',
    'THE DELIVERY SYSTEM: ADMINISTRATIVE SUPERVISION; NOTICE REQUIREMENTS; AUDIT PROCEDURES; ANTI-FRAUD LEGISLATION; INJURY PREVENTION PROGRAMS; INFORMATION AND ASSISTANCE PROGRAM; NEW SB 863 ORGANIZATIONS': 'delivery_system',
    'SETTLEMENTS: Compromise and Release, and Stipulated Findings & Award': 'settlements',
    'CLAIM FILING PROCEDURE, PLEADINGS, VENUE AND DISMISSAL': 'claim_filing',
    'ATTORNEYS AND REPRESENTATIVES': 'attorneys',
    'STATUTE OF LIMITATIONS: Original Filings and Reopenings': 'statute_limitations',
    'PREPARATION FOR TRIAL AND OTHER PRE-TRIAL MATTERS': 'trial_prep',
    'PRE-TRIAL DISCOVERY: Depositions and Other Techniques': 'pretrial_discovery',
    'TRIAL SETTING AND TRIAL': 'trial',
    'JUDGMENTS: Post-Trial Dispositions, Decisions and Other Matters (Interest, Costs, Credit, Clerical Error, Enforcement, Commutation and Restitution)': 'judgments',
    'RECONSIDERATION AND REVIEW': 'reconsideration',
    'THIRD-PARTY SUITS': 'third_party_suits'
}


import PyPDF2
import tqdm
import os

def split_pdf(input_pdf_path, chapter_dict, filename_dict, output_dir):
    """
    Split a PDF into multiple parts based on specified page ranges.

    Parameters:
        input_pdf_path (str): Path to the input PDF file.
        chapter_dict (dict): Dictionary mapping output file names to page ranges.
        filename_dict (dict): Dictionary mapping output file names to desired file names.
        output_dir (str): Directory where output PDF files will be saved.
    """
    # Open the input PDF
    with open(input_pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Loop through specified page ranges and output paths
        for output_file, (start, end) in tqdm.tqdm(chapter_dict.items(), total=len(chapter_dict.keys())):
            # Create a PDF writer for each range
            pdf_writer = PyPDF2.PdfWriter()

            # Add specified pages to the writer
            for page_num in range(start - 1, end):  # PyPDF2 uses 0-based index
                pdf_writer.add_page(pdf_reader.pages[page_num])

            # Add .pdf extension to the output file name
            output_path = os.path.join(output_dir, f"{filename_dict[output_file]}.pdf")

            # Write the output PDF file
            with open(output_path, "wb") as output_pdf:
                pdf_writer.write(output_pdf)

            print(f"Saved split PDF to: {output_path}")

input_pdf_path = 'book.pdf'
output_path = 'data'
split_pdf(input_pdf_path, chapter_dict,chapter_filenames, output_path)

chapter_docs = {}
for chapter_name , chapter_filename in tqdm.tqdm(chapter_filenames.items(), total = len(chapter_dict.keys())):
    chapter_docs[chapter_name] = SimpleDirectoryReader(
        input_files=[f"data/{chapter_filename}.pdf"]
    ).load_data()


Settings.llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

node_parser = SentenceSplitter()

# Build agents dictionary
agents = {}
query_engines = {}

# this is for the baseline
all_nodes = []
legal_assistant = {}

for chapter_name , chapter_filename in tqdm.tqdm(chapter_filenames.items(), total = len(chapter_filenames.keys())):
    nodes = node_parser.get_nodes_from_documents(chapter_docs[chapter_name])
    all_nodes.extend(nodes)

    if not os.path.exists(f"data/{chapter_filename}"):
        # build vector index
        vector_index = VectorStoreIndex(nodes)
        vector_index.storage_context.persist(
            persist_dir=f"data/{chapter_filename}"
        )
    else:
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=f"data/{chapter_filename}"),
        )

    # build summary index
    summary_index = SummaryIndex(nodes)
    # define query engines
    vector_query_engine = vector_index.as_query_engine(llm=Settings.llm)
    summary_query_engine = summary_index.as_query_engine(llm=Settings.llm)

    # define tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_tool",
                description=(
                  "Optimized for detailed inquiries related to specific sections within"
                  f" {chapter_name}. Use this for questions targeting aspects such as"
                  " definitions, laws, and case studies related to each chapter, covering"
                  " topics like employment, disabilities, insurance, and legal procedures."
              ),
            ),
        ),
        QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name="summary_tool",
                description=(
                  "Ideal for comprehensive overviews and summaries across"
                  f" all sections of {chapter_name}. Use this tool for a holistic"
                  " perspective or for answering general questions that require"
                  " insights from multiple chapters without focusing on one section."
              ),
            ),
        ),
    ]

    # build agent
    function_llm = OpenAI(model="gpt-4")
    agent = OpenAIAgent.from_tools(
        query_engine_tools,
        llm=function_llm,
        verbose=True,
        system_prompt=f"""\
          You are a specialized agent designed to answer queries about {chapter_name}.
          You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
          """,
          )

    agents[chapter_filename] = agent
    query_engines[chapter_filename] = vector_index.as_query_engine(
        similarity_top_k=2
    )


# define tool for each document agent
all_tools = []
for chapter_name , chapter_filename in tqdm.tqdm(chapter_filenames.items(), total = len(chapter_filenames.keys())):
    chapter_summary = (
        f"This content contains chapters  about {chapter_name}. Use"
        f" this tool if you want to answer any questions about {chapter_name}.\n"
    )
    doc_tool = QueryEngineTool(
        query_engine=agents[chapter_filename],
        metadata=ToolMetadata(
            name=f"tool_{chapter_filename}",
            description=chapter_summary,
        ),
    )
    all_tools.append(doc_tool)

# define an "object" index and retriever over these tools
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)

system_prompt = """\
You are an advanced agent assigned to provide precise, reliable answers based on a structured set of legal document chapters.
Your role is to navigate complex, chapter-specific content to give clear, contextually accurate responses for legal inquiries.

Guidelines for your responses:
1. **Use Designated Tools Only**: Draw all information exclusively from the tools provided, without relying on external or prior knowledge. Each response should be strictly based on the content retrieved from the designated sections of the legal document.

2. **Organize Responses by Legal Context**:
   - For each question, determine the relevant chapter(s) within the legal document to ensure responses are aligned with the legal concepts in those sections.
   - Address core aspects of the legal topics (e.g., jurisdiction, employment relationship, benefits, statutes of limitations) with clarity and precision.

3. **Interpret Legal Queries Effectively**:
   - Identify the query’s intent—is it seeking an overview, specific legal definitions, procedural guidance, or case-based examples?
   - Tailor responses to reflect the legal language and terminology of the document, focusing on concise yet comprehensive answers suitable for legal reference.

4. **Adapt to Query Type**:
   - For definitions or statutory references (e.g., "Define jurisdiction in this context"), extract and clarify legal terms directly from the document.
   - For procedural or rule-based inquiries (e.g., "What are the steps for filing a claim?"), structure answers in sequential or bullet format to provide clear, actionable guidance.
   - For comparative questions (e.g., "How does temporary disability differ from permanent disability?"), highlight critical distinctions relevant to the legal framework.

5. **Respond in Clear, Legal Language**:
   - Use precise legal terms where relevant, ensuring clarity for legal professionals. If required, break down complex concepts into structured sections.
   - Maintain formality and professionalism in tone, presenting information as would be expected in a legal reference.

6. **Acknowledge Information Limits**:
   - If the tools return limited information on a topic, indicate this clearly. For example, if a query falls outside the document’s coverage, politely note the limitation.

Your objective is to serve as a dependable, tool-driven resource, assisting in navigating complex legal material with integrity and accuracy. Follow these guidelines to provide responses aligned with each specific chapter and section of the legal document.
"""

from llama_index.core.agent import ReActAgent

top_agent = ReActAgent.from_tools(
    tool_retriever=obj_index.as_retriever(similarity_top_k=3),
    system_prompt=system_prompt,
    verbose=True,
)
from typing import List, Dict, Any
import autogen
from autogen import Agent, AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

base_index = VectorStoreIndex(all_nodes)
base_query_engine = base_index.as_query_engine(similarity_top_k=4)

class LegalGroupChat:
    def __init__(self, top_agent, config_list):
        self.top_agent = top_agent
        self.config_list = config_list
        self.setup_agents()
        self.create_groupchat()
    
    def setup_agents(self):
        # Context Analysis Agent
        self.context_agent = AssistantAgent(
            name="Context_Analyzer",
            system_message="""You are a specialized legal context analyzer. Your role is to:
            1. Analyze questions to determine which legal areas and chapters are most relevant
            2. Provide necessary background context for legal questions
            3. Help frame complex legal queries in the most effective way
            4. Work with the Legal_Researcher to ensure all relevant context is considered
            
            After analyzing, pass your findings to the Legal_Researcher.""",
            llm_config={"config_list": self.config_list}
        )

        # Legal Research Agent (integrated with top_agent)
        self.research_agent = AssistantAgent(
            name="Legal_Researcher",
            system_message="""You are a legal research assistant with access to specialized legal documents. Your role is to:
            1. Use the research_legal_documents function to query the knowledge base
            2. Analyze and synthesize the information received
            3. Collaborate with other agents to ensure comprehensive coverage
            4. Always provide specific citations and references
            
            After research, pass your findings to the Legal_Writer.""",
            llm_config={"config_list": self.config_list}
        )

        # Legal Writing Agent
        self.writer_agent = AssistantAgent(
            name="Legal_Writer",
            system_message="""You are a specialized legal writer. Your responsibilities include:
            1. Converting complex legal information into clear, precise language
            2. Structuring responses in a professional legal format
            3. Ensuring all citations and references are properly formatted
            4. Maintaining consistent legal terminology
            5. Creating well-organized summaries and explanations
            
            After writing, pass your draft to the Legal_Reviewer.""",
            llm_config={"config_list": self.config_list}
        )

        # Review Agent
        self.review_agent = AssistantAgent(
            name="Legal_Reviewer",
            system_message="""You are a critical legal reviewer. Your duties are to:
            1. Verify accuracy of legal interpretations
            2. Check for completeness of responses
            3. Identify potential missing elements or considerations
            4. Ensure compliance with legal standards
            5. Provide the final decision or recommendation
            
            Conclude with a clear "FINAL DECISION:" section that summarizes the key points and recommendations.""",
            llm_config={"config_list": self.config_list}
        )

    def create_groupchat(self):
        # Define the group chat with sequential flow
        self.agents = [
            self.context_agent,
            self.research_agent,
            self.writer_agent,
            self.review_agent
        ]

        self.group_chat = GroupChat(
            agents=self.agents,
            messages=[],
            speaker_selection_method="round_robin",  # Ensure orderly progression
            allow_repeat_speaker=False  # Prevent same speaker twice in a row
        )

        # Create the group chat manager
        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config={"config_list": self.config_list}
        )

    def initiate_chat(self, message: str):
        """Start a new group chat discussion with sequential agent interaction"""
        def research_function(message: str) -> str:
            try:
                response = self.top_agent.query(message)
                if not response:
                    return "No specific information found. Please try rephrasing the question."
                return str(response)
            except Exception as e:
                return f"Error in research: {str(e)}"

        # Register the research function with the research agent
        self.research_agent.register_function(
            function_map={
                "research_legal_documents": research_function
            }
        )

        # Create a structured initial message with clear workflow
        initial_message = {
            "content": f"""LEGAL QUERY: {message}
            
            WORKFLOW:
            1. Context_Analyzer: Analyze the legal context and relevant areas
            2. Legal_Researcher: Query knowledge base and synthesize findings
            3. Legal_Writer: Format and structure the response
            4. Legal_Reviewer: Review and provide final decision
            
            Please proceed with the analysis.""",
            "role": "user"
        }

        # Start the chat with the context analyzer
        try:
            self.context_agent.initiate_chat(
                self.manager,
                message=initial_message,
                clear_history=False
            )
        except Exception as e:
            print(f"Error initiating chat: {str(e)}")
            raise

def create_group_chat(top_agent, openai_api_key: str):
    """Create a new legal group chat instance with error handling"""
    try:
        config_list = [
            {
                "model": "gpt-4",
                "api_key": openai_api_key,
            }
        ]
        return LegalGroupChat(top_agent, config_list)
    except Exception as e:
        print(f"Error creating group chat: {str(e)}")
        raise

# Usage
group_chat = create_group_chat(
    top_agent,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
    
# Example query
question = "Can you analyze the requirements for filing a workers' compensation claim and explain the statute of limitations?"
group_chat.initiate_chat(question)

