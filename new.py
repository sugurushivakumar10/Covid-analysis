from flask import Flask, render_template, request, jsonify
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
import pandas as pd
import pyodbc
from sqlalchemy import create_engine
import urllib.parse
import openai
import json

openai_api_key = "openapikey"

qa_dict = {}
qa_dict_file = "qa_dict.json"

class ChatBot:
    def __init__(self):
        self.history = []

    def add_to_history(self, user_input, model_response):
        self.history.append({"user": user_input, "bot": model_response})

    def get_context(self):
        context = ""
        for exchange in self.history[-5:]:
            context += f"User: {exchange['user']}\nBot: {exchange['bot']}\n"
        return context

    def generate_response(self, user_input):
        context = self.get_context()
        full_input = context + f"User: {user_input}\nBot:"
        model_response = self.mock_model_response(full_input)
        self.add_to_history(user_input, model_response)
        return model_response

    def mock_model_response(self, input_text):
        # Replace with actual model logic or API call
        return "This is a simulated response based on the context."

def save_qa_dict(qa_dict_file):
    with open(qa_dict_file, 'w') as f:
        json.dump(qa_dict, f, indent=4)

def load_data_from_sql():
    conn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};Server=SHASHA_SHAAN;Database=VoiceAssistance;Trusted_Connection=yes;')
    query = "SELECT * FROM ToyotaCorolla"
    data_sql = pd.read_sql(query, conn)
    conn.close()
    return data_sql

data = load_data_from_sql()
agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=openai_api_key), data, verbose=True, return_intermediate_steps=True)

app = Flask(__name__)
chatbot = ChatBot()  # Create an instance of the ChatBot

@app.route("/")
def index():
    return render_template('chat3.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print("Input is ===>", msg)
    return get_chat_response(msg)

def get_chat_response_sql(text):
    schema_info = """
    [ToyotaCorolla]
    [Model] [varchar](max) NULL,
    [Price] [bigint] NULL,
    [Age_08_04] [bigint] NULL,
    [Mfg_Month] [bigint] NULL,
    [Mfg_Year] [bigint] NULL,
    [KM] [bigint] NULL,
    [Fuel_Type] [varchar](max) NULL,
    [HP] [bigint] NULL,
    [Met_Color] [bigint] NULL,
    [Color] [varchar](max) NULL,
    [Automatic] [bigint] NULL,
    [cc] [bigint] NULL,
    [Doors] [bigint] NULL,
    [Cylinders] [bigint] NULL,
    [Gears] [bigint] NULL,
    [Quarterly_Tax] [bigint] NULL,
    [Weight] [bigint] NULL,
    [Mfr_Guarantee] [bigint] NULL,
    [BOVAG_Guarantee] [bigint] NULL,
    [Guarantee_Period] [bigint] NULL,
    [ABS] [bigint] NULL,
    [Airbag_1] [bigint] NULL,
    [Airbag_2] [bigint] NULL,
    [Airco] [bigint] NULL,
    [Automatic_airco] [bigint] NULL,
    [Boardcomputer] [bigint] NULL,
    [CD_Player] [bigint] NULL,
    [Central_Lock] [bigint] NULL,
    [Powered_Windows] [bigint] NULL,
    [Power_Steering] [bigint] NULL,
    [Radio] [bigint] NULL,
    [Mistlamps] [bigint] NULL,
    [Sport_Model] [bigint] NULL,
    [Backseat_Divider] [bigint] NULL,
    [Metallic_Rim] [bigint] NULL,
    [Radio_cassette] [bigint] NULL,
    [Tow_Bar] [bigint] NULL
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=300,
        temperature=0.6,
        n=1,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that converts natural language to SQL queries."},
            {"role": "system", "content": f"Here is the schema information for the database: {schema_info}"},
            {"role": "user", "content": f"Convert the following natural language description into an SQL query: \n\n{text}\n\nSQL Query"}
        ],
    )
    
    matched_answer = response.choices[0].message['content']
    user_query = matched_answer.replace("\n", " ").split(";")[0]
    return user_query

def update_stored_procedure(sql_query):
    conn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};Server=SHASHA_SHAAN;Database=VoiceAssistance;Trusted_Connection=yes;')
    cursor = conn.cursor()
    try:
        drop_query = """
        IF EXISTS (
            SELECT *
            FROM sys.objects
            WHERE object_id = OBJECT_ID(N'[dbo].[SelectFromStagingTable]') AND type IN (N'P', N'PC')
        )
        DROP PROCEDURE [dbo].[SelectFromStagingTable];
        """
        cursor.execute(drop_query)
        conn.commit()

        create_query = f"""
        CREATE PROCEDURE [dbo].[SelectFromStagingTable]
        AS
        BEGIN
            SET NOCOUNT ON;
            -- User-provided SQL query
            {sql_query}
        END
        """
        cursor.execute(create_query)
        conn.commit()

        print("Stored procedure updated successfully.")
    except pyodbc.Error as e:
        print(f"Error updating stored procedure: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()

def get_chat_response(text):
    try:
        # Handle the chatbot response
        chatbot_response = chatbot.generate_response(text)

        qa_dict[text] = None
        agent_response = agent({"input": text})
        answer = agent_response
        print("answer=========>", answer)
        text_answer = answer['output']
        qa_dict[text] = text_answer
        print("text_answer=========>", text_answer)

        sql_query_response = get_chat_response_sql(text)
        sql_query_response = str(sql_query_response)
        qa_dict[text] = sql_query_response
        save_qa_dict(qa_dict_file)
        print("SQL query response after response ===>", sql_query_response)

        if "SELECT *" in sql_query_response:
            update_stored_procedure(sql_query_response)
            text_answer = "Refreshed and presented visually"
        
            text_df = pd.DataFrame({text_answer: [text_answer]})
            text_df.columns = ['Output of question is']

            params = urllib.parse.quote_plus("DRIVER={ODBC Driver 17 for SQL Server};SERVER=SHASHA_SHAAN;DATABASE=VoiceAssistance;Trusted_Connection=yes")
            engine = create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))

            text_df.to_sql('txtstagingtable', schema='dbo', con=engine, chunksize=50, method='multi', index=False, if_exists='replace')

            return str(sql_query_response)
        else:
            text_df = pd.DataFrame({text_answer})
            text_df.columns = ['Output of question is']

            params = urllib.parse.quote_plus("DRIVER={ODBC Driver 17 for SQL Server};SERVER=SHASHA_SHAAN;DATABASE=VoiceAssistance;Trusted_Connection=yes")
            engine = create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))

            text_df.to_sql('txtstagingtable', schema='dbo', con=engine, chunksize=50, method='multi', index=False, if_exists='replace')
            sql_query = "Select * from "
            update_stored_procedure(sql_query)
            return text_answer
    except BaseException as e:
        return f'Failed to do something: {str(e)}'

if __name__ == '__main__':
    app.run(debug=True)  # Enable debug mode for development
