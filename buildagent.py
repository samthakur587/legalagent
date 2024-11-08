import streamlit as st
import dill as pickle
from autogen import Agent, AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

import streamlit as st

class TrackableAssistantAgent(AssistantAgent):
    def _process_received_message(self, message, sender, silent):
        # Alternate message alignment based on message count
        message_index = st.session_state.get("message_count", 0)
        align_side = "flex-start" if message_index % 2 == 0 else "flex-end"
        message_background = (
            "linear-gradient(135deg, #4a90e2, #6a4e9d)"
            if message_index % 2 == 0
            else "linear-gradient(135deg, #2dbdb3, #5b78e5)"
        )


        # Display the assistant message
        with st.chat_message("assistant"):
            st.markdown(
                f"""
                <div style='display:flex;align-items:center;justify-content:{align_side};margin:10px 0;'>
                    <div style="background:{message_background};padding:12px 16px;border-radius:12px;max-width:75%;">
                        <div style='font-weight:bold;color:black;text-align:left;'>{message.get('name')}</div>
                        <div style='text-align:left;color:#f5f5f5;'>{message.get("content")}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        # Update the message count
        st.session_state["message_count"] = message_index + 1
        return super()._process_received_message(message, sender, silent)

class TrackableUserProxyAgent(UserProxyAgent):
    def _process_received_message(self, message, sender, silent):
        # Alternate message alignment based on message count
        message_index = st.session_state.get("message_count", 0)
        align_side = "flex-end" if message_index % 2 == 0 else "flex-start"
        message_background = "linear-gradient(135deg, #994c00, #ff9800)" if message_index % 2 == 0 else "linear-gradient(135deg, #e68a00, #ffb74d)"

        # Display the user message
        with st.chat_message("user"):
            st.markdown(
                f"""
                <div style='display:flex;align-items:center;justify-content:{align_side};margin:10px 0;'>
                    <div style="background:{message_background};padding:12px 16px;border-radius:12px;max-width:60%;">
                        <div style='font-weight:bold;color:white;text-align:left;'>User</div>
                        <div style='text-align:left;color:#f5f5f5;'>{message.get("content")}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        # Update the message count
        st.session_state["message_count"] = message_index + 1
        return super()._process_received_message(message, sender, silent)


class LegalAgent:
    def __init__(self, data_dir='data'):
        with open(f"agent_setup.pkl", "rb") as f:
            setup = pickle.load(f)
            self.agents = setup['agents']
            self.query_engines = setup['query_engines']
            self.top_agent = setup['top_agent']

    def query(self, question):
        return self.top_agent.query(question)

class LegalGroupChat:
    def __init__(self, top_agent, config_list):
        self.top_agent = top_agent
        self.config_list = config_list
        self.setup_agents()
        self.create_groupchat()

    def setup_agents(self):
        # Context Analysis Agent
        self.context_agent = TrackableAssistantAgent(
            name="Context_Analyzer",
            system_message="""You are a specialized legal context analyzer. Your primary goal is to:
            1. Identify and clarify the specific legal domains relevant to the question (e.g., business law, intellectual property).
            2. Highlight any additional background context that will assist other agents.
            3. Recommend key sections or documents to consult.
            Output should include a summary of legal areas and related guidance.
            """,
            llm_config={"config_list": self.config_list}
        )

        # Legal Research Agent (integrated with top_agent)
        self.research_agent = TrackableAssistantAgent(
            name="Legal_Researcher",
            system_message="""You are a legal research assistant with access to a specialized knowledge base. Your task is to:
            1. Use the research_legal_documents function to retrieve detailed legal insights based on Context_Analyzer recommendations.
            2. Prioritize primary legal resources, statutes, and case law relevant to the query.
            3. Return organized findings with citations and short interpretations.
            """,
            llm_config={"config_list": self.config_list}
        )

        # Legal Writing Agent
        self.writer_agent = TrackableAssistantAgent(
            name="Legal_Writer",
            system_message="""You are a legal document writer tasked with creating a structured response. Your job is to:
            1. Organize findings from Legal_Researcher into a cohesive format.
            2. Write in a clear, concise legal tone with proper terminology.
            3. Structure the response to address each part of the query logically.
            4. as the legal write you will be finalizing the response and preparing it for review and final decision.
            """,
            llm_config={"config_list": self.config_list}
        )

        # Review Agent
        self.review_agent = TrackableAssistantAgent(
            name="Legal_Reviewer",
            system_message="""You are a critical reviewer who ensures legal accuracy. Your responsibilities include:
            1. Reviewing the draft for accuracy and completeness.
            2. Ensuring adherence to legal standards and making necessary corrections.
            3. Providing a final recommendation with a "FINAL DECISION:" summary that addresses the main query directly.
            """,
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
            speaker_selection_method="round_robin",
            allow_repeat_speaker=False
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
                print(f"Research response: >>> {response}")
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
            1. Context_Analyzer: Identify relevant legal areas and any required context.
            2. Legal_Researcher: Retrieve and synthesize details from the knowledge base.
            3. Legal_Writer: Format the response in a clear legal document style.
            4. Legal_Reviewer: Verify and provide a final recommendation in the "FINAL DECISION:" section.
            
            Please proceed with the analysis.""",
            "role": "user"
        }

        # Start the chat with the context analyzer
        try:
            self.context_agent.initiate_chat(
                self.manager,
                message=initial_message,
                clear_history=True
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

def main():
    st.title("Legal Query Assistant")

    with st.sidebar:
        st.write("OpenAI Configuration")
        api_key = st.text_input("API Key", type="password")


    if api_key:
        top_agent = LegalAgent().top_agent
        group_chat = create_group_chat(top_agent, api_key)

        user_input = st.chat_input("Enter your legal question:")

        if user_input:
            chat_container = st.empty()
            group_chat.initiate_chat(user_input)

            for message in group_chat.group_chat.messages:
                with chat_container.container():
                    st.subheader(f"{message['role'].capitalize()} Response:")
                    st.markdown(message['content'])

if __name__ == "__main__":
    main()