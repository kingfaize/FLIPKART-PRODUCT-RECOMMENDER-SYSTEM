from langchain_groq import ChatGroq
from langchain.agents import create_agent
from flipkart.config import Config

class RAGChainBuilder:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.model = ChatGroq(model=Config.RAG_MODEL, temperature=0.5)

    def build_chain(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        def retrieve_tool(query: str) -> str:
            """Retrieve relevant product reviews and context for a given query."""
            docs = retriever.invoke(query)
            return "\n".join([doc.page_content for doc in docs])

        agent = create_agent(
            model=self.model,
            tools=[retrieve_tool],
            system_prompt=(
                "You're an e-commerce bot answering product-related queries using reviews and titles. "
                "Stick to context. Be concise and helpful."
            )
        )
        return agent

    def invoke_with_history(self, agent, user_input, chat_history=None):
        # chat_history should be a list of dicts: [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}, ...]
        if chat_history is None:
            chat_history = []
        messages = chat_history + [{"role": "user", "content": user_input}]
        response = agent.invoke({"messages": messages})
        return response



