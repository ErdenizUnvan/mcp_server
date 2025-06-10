#capitalize words for this sentence: what is your name?
#count how many words are there in the sentence of "what is your name?"
#get the wheather temperature at the city of Istanbul
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_DIR = r"C:\Users\Dell\llama3_2\mcp\project\facebook-bart-large-mnli"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

classifier = pipeline(
    "zero-shot-classification",
    model=model,
    tokenizer=tokenizer,
    hypothesis_template="This request is about {}.",
)

candidate_labels = [
    "count words",
    "capitalize words",
    "get weather temperature of the city",
    "other"
]

system_prompt = (
    "You are a function-calling AI agent.\n"
    "The function you need to call will be selected according to the user's input for you and it will be informed to you\n"
    "Only provide the outcome of the function, do not make any additonal comments or statements\n"
)

chat_model  = ChatOllama(model="llama3.2").with_config({"system_prompt": system_prompt})

async def main():
    while True:
        user_input = input("\n🧠 Query (exit to quit): ")
        user_input=user_input.lower()

        if user_input.lower() in ['quit','exit','bye']:
            break

        result = classifier(user_input, candidate_labels)
        top_label = result['labels'][0]
        top_score = result['scores'][0]
        THRESHOLD = 0.65
        if top_label == 'other' or top_score < THRESHOLD:
            print('Out of scope as function!')
        else:
            if top_label=='count words':
                top_label='count_words'
            if top_label=='capitalize words':
                top_label='capitalize_words'
            if top_label=='get weather temperature of the city':
                top_label='get_weather_temperature'
            print(f'top_label:{top_label}')

            server_params = StdioServerParameters(
                command="python",
                args=["test_tools_mcp_server.py"],
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await load_mcp_tools(session)
                    selected_tools = [tool for tool in tools if tool.name==top_label]
                    agent = create_react_agent(chat_model, selected_tools)

                    msg = {"messages": user_input}
                    response = await agent.ainvoke(msg)

                    print("\n🧾 Response:")
                    tool_output_found = False
                    for m in response["messages"]:
                        if m.type == "tool":
                            print(m.content)
                            tool_output_found = True

                    if not tool_output_found:
                        print("No function output was returned.")

if __name__ == "__main__":
    asyncio.run(main())
