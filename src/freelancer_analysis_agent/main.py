import os

import pandas as pd
import click
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

def create_agent(file_path: str, llm):
    """
    Загружает данные и создает Pandas DataFrame агент.
    """
    try:
        print(f"Загрузка данных из '{file_path}'...")
        df = pd.read_csv(file_path)
        # Заменяем пробелы в названиях колонок для удобства LLM
        df.columns = df.columns.str.replace(' ', '_')
        print("Данные успешно загружены.")
    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути '{file_path}'.")
        print("Убедитесь, что файл существует и путь указан верно.")
        return None

    print("Создание агента...")
    
    AGENT_PREFIX = """
You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
You MUST use this dataframe `df` to answer the user's questions.
Do NOT create your own dataframes.
"""

    AGENT_SUFFIX = """
Begin!
Question: {input}
Thought: {agent_scratchpad}
Remember to finalize your answer in Russian and make it clear and easy to understand for the user.
"""

    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        agent_executor_kwargs={"handle_parsing_errors": True},
        allow_dangerous_code=True,
        prefix=AGENT_PREFIX,
        include_df_in_prompt=None,
        suffix=AGENT_SUFFIX
    )
    print("Агент успешно создан.")
    return agent

@click.command()
@click.argument('question', type=str)
def cli(question: str):
    """
    AI-ассистент для анализа данных о фрилансерах.

    Задайте ему QUESTION на естественном языке о данных в CSV файле.
    Пример:
    python main.py "Какой средний доход у фрилансеров из США?"
    """
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Ошибка: Не найден GOOGLE_API_KEY в .env файле.")
        return

    # Используем корректное и стабильное имя модели
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    except:
        # Fallback модель
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    agent = create_agent("data/freelancer_earnings_bd.csv", llm)
    
    if agent is None:
        return

    print("-" * 30)
    print(f"Ваш вопрос: {question}")
    print("...Агент думает...")
    print("-" * 30)

    try:
        response = agent.invoke({"input": question})
        answer = response["output"]
    except Exception as e:
        answer = f"Произошла ошибка при выполнении запроса: {e}"

    print("-" * 30)
    print(f"Ответ: {answer}")


if __name__ == "__main__":
    cli()
