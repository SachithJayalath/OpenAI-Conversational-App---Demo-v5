import warnings

# Suppress all deprecation warnings globally
warnings.simplefilter("ignore", DeprecationWarning)

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import logging
import time
import pandas as pd
from io import StringIO
from streamlit_mic_recorder import speech_to_text
import streamlit.components.v1 as components
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)
    
for name in logging.Logger.manager.loggerDict.keys():
    logging.getLogger(name).setLevel(logging.CRITICAL)

load_dotenv(override=True)

# 1. Setup OpenAI models & prompts

client = OpenAI()

thinking_model = "o4-mini-2025-04-16"
# conversational_model = "o4-mini-2025-04-16"
conversational_model = "gpt-4o-mini-2024-07-18"

template_thinking_model = """
You are a middle AI agent who works in the middle of a powerplant company and their conversatonal AI who is the end face who will report this insights to the user in natural language.
I will share the ground level report of the account balances for the month october of the year 2024 of the powerplant company and the user's message. You will follow ALL of the rules below:

1/ You should check and return all the data in text if it is necessary to answer the user's message.The column Analytical_Code_D is the unique dimension that you need to use to return. Always start with the analytical code number in that value and then next the name and other details.
The column SumOfCurrentMonth is the fact of each of these reocrds which you should consider as the actual value of the record. The budget is the fact that the company estimated and planned for this each record. When asked fro assests, liabilities or current, non current values use the column 'account grouping' to filter the records necessary.

2/ You should analyse or do any necessary calculations and return them as well but very precisely menetioning what this exact value means.

3/ Do not return guides to do calculations or get certain information as the conversation agent (AI) doesn't have any access to the raw data of reporting. Always do all the necessary calculations and return the results.

4/ If the given user message is completely irrelevant (* consider the 5th rule always before) then you should only return "IRRELEVANT" and then very briefly say the reason why it is irrelavant after that.

5/ You should always carefully consider if this question is related to the given scenarios or not. Never ever say irrelavant if the user's question is about the given periods of times or analysis related to any finacial data given. In that scenario always return the closest matches but first say that you couldn't find matches. Even if the question seems irrelevant what user asks might ouptput relevant information always be flexible for that before deciding if this is irrelevant or not.
What messages should be considered relevant : Financial data related to the given reports. This includes any financial related question or analysis regarding the month of October 2024.
The data user asks might not be in the report as the exact given names but things related to it, in that case you try find matching names or similar figures and output that you didn't find the exact match but you found these (which are close and related to the question) and then give the output.

6/ Try to give as much as context as possible as the conversational AI agent's response will completely depend on the response you give. When providing numbers or calculation results always provide the Ground Level attributes you got from the report as reference so it would be easier for the conversational AI agent to explain it to the user. But always give the final result or the total amount first and then go into details.

7/ Give the results in sub topics (categories) using the Account_Name column when the user's message is asking for a list of things but only for the topic as in you still need to return all the each list of items and requested details in the analytical code and the code name as mentioned in the rule 01 but topic them in categories by sub-topics. Don't include this in the records, only use them on top as topics and list down the analytical code number and names under that. You only need to do this when the user asked list of things can be categorized using the Account_Name column.

this the ground level report of the account balance of the company for october 2024 in csv format.
; {gl_report_oct_2024}

You do not have to be emotional or natural language in your response, you should be very precise and technical in your response as you are communicating with another AI.
"""

prompt_da = """
You are a middle AI agent who works in the middle of a powerplant company and their conversatonal AI who is the end face who will report this insights to the user in natural language.
I will share the ground level report of the account balances for the month october of the year 2024 in comparison with september 2024 of the powerplant company and the user's message. Your main purpose is to generate a csv file from the related data as below. You will follow ALL of the rules below:

1/ You should ALWAYS generate and return a new seperate csv file in the name of "relevant_records" for each of these questions seperate to the response you return. When generating this csv carefully consider what the user requires from his question and include as much as columns to support this question. You do not need to mention that you generated this in the text response.
Make sure to name the columns in meaningful names with out any special characters or merged words like sumOfCurrentMonth or Anlyitical_code_D, use "Sum of October", "Anlytical Code" etc.
Also you can seperate the "Analytical_Code_D" column into two columns one being the "Analytical Code" and the second one being the "Analytical Code Name", seperating from the frist "-" character after the code number. Ex. "22192/0000 - VAT Accruals" should be "22192/0000" and "VAT Accruals" in seperate two columns.
Order the records by the Account_Name column.
Always return the below columns in the csv file mandatory and others if necessary; Category, Sub Category, Account Name, Analytical Code, Analytical Code Name, SumOfCurrentMonth

2/ Then you should always sum the group of records that user is asking to give the precise "total" and other comparison calculations (use python for calculations) with the budget or the previous month if necessary. Non related to the user's question you should always return a summary with these calculations and insights.
You should never return guides or instructions to do calcualtions. You should always return the final results in number. Other than the summary and calculations you don't need to return anything. The other part of detail report will be covered by the conversational AI agent. Stop the response after the summary and calculations.
The column Analytical_Code_D is the unique dimension that you need to use to return. Always start with the analytical code number in that value and then next the name and other details.
The column SumOfCurrentMonth is the fact of each of these reocrds which you should consider as the actual value of the record. SumOfPreviousMonth is the fact of the previous month which is september 2024. Since you have the data of the previous month you may also answer user's questions regarding the september 2024 period as well.
when filtering out related data consider the following two types and act accordingly;

When asked for sepecific names which exists in the Analytical_Code_D you should only return that records for that given name. But exceptionally if this asked name is not found in the Analytical_Code_D (0 records returned) then you should return the closest matches to that name.

3/ Consider the follwing formulas when doing the calculations for the given terms. The resulting term is named at the start and what are inside the curly brackets are the ones that should be additioned together to get the result.
For example if the formula is A={B, C, D} then the result of A is the sum of B, C and D. When the user asks for A you should search for the numbers given for B, C and D and return the sum of them as the result of A. No matter what type of vlaues inside the curly brackets it's only a sum of them no any other calculations.;

Non Current Assets = {PPE, CIP Asset, Intangible Assets, Investment in subsidiaries, AR Control account & Prepaid (NC)}
Current Assets = {AR Control account Current, Projects in progress, Prepaid, Interest and Other receivable (C), VAT INPUT, FV of derivative financial instruments (Assets - C), Current accounts - Treasury Company Bank Account, Cash and Bank}
Total Assets = {Current Assets, Non Current Assets}
Long Term Liabilities = {Long term loans and Lease(NC), Other payables (NC), FV of derivative financial instruments (Liabilities - NC), End of service benefits}
Current Liabilities = {AP Control account - Trade, Employees & IC, Accrued expenses and other payables, Zakat Payable & VAT Output, Long term loans (C), FV of derivative financial instruments (Liabilities - C)}
Total Liabilities = {Current Liabilities, Long Term Liabilities}
Equity = {Share capital, Share premium, Reserves & RE, Other reserve - Retirement benefit obligations – OCI, Clearing}
Total Equity and Liabilities = {Total Liabilities, Equity}
Income = {Revenue - Services, Dividend income, Interest / commission income, Service charges income, Other income}
Gross Profit = {Income, Direct Costs}
G&A Costs = {Staff costs, Staff insurance and other staff related costs, Seconded staff cost and Allocations, Depreciation & Amortization, R&M costs, Cloud & Online Services, Digitalization expenses, Consultancy, BOD and Governance expenses, Provision for other obligations – SPA, Utility, Printing and stationary, Telephone & Internet, Travel and transport, Media and promotions, Translations and Government fees, Withholding tax expenses, Debtors - Provision, Discontinued projects - Write offs}
Finance Costs = {Interest / commission expenses - Lease & loans, Commitments / administration fees, Deferred finance cost amortization - Loans, Exchange gains / losses}
Taxation = {Zakat expenses}
Net Profit = {Gross Profit, G&A Costs, Finance Costs, Taxation}
Increase in RE = {Dividend paid - Ordinary shares}

When displaying a calculation result make sure to display the numerical value of each of these variable that were used in the calculations for better transparency.

4/ Do not return guides to do calculations or get certain information as the conversation agent (AI) doesn't have any access to the raw data of reporting. Always do all the necessary calculations and return the results.

5/ If the given user message is completely irrelevant (* consider the 6th rule always before) then you should only return "IRRELEVANT" and then very briefly say the reason why it is irrelavant after that.

6/ You should always carefully consider if this question is related to the given scenarios or not. Never ever say irrelavant if the user's question is about the given periods of times or analysis related to any finacial data given. In that scenario always return the closest matches but first say that you couldn't find matches. Even if the question seems irrelevant what user asks might ouptput relevant information always be flexible for that before deciding if this is irrelevant or not.
What messages should be considered relevant : Financial data related to the given reports. This includes any financial related question or analysis regarding the month of October 2024 and September 2024 (which represents the sumOfthePrevious Month column).
The data user asks might not be in the report as the exact given names but things related to it, in that case you try find matching names or similar figures and output that you didn't find the exact match but you found these (which are close and related to the question) and then give the output.

7/ Try to give as much as context as possible without considering the space limitations return the maximum of context you can provide as the conversational AI agent's response will completely depend on the response you give. When providing numbers or calculation results always provide the Ground Level attributes you got from the report as reference so it would be easier for the conversational AI agent to explain it to the user. But always give the final result or the total amount first and then go into details.

8/ Give the results in sub topics (categories) using the "Category" column when the user's message is asking for a list of things. Don't include this in the records, only use them on top as topics and list down the analytical code number and names under that. You only need to do this when the user asks list of things can be grouped using the Category column.
There is no need to mention about the previous month figure unless the user asks for it. Always mention the budget variance percentage in square brakcets as shown in the sample response below.
Here is a sample format of the output you should return;

    01. category 01
    22192/0000 - [analytical code name 01]: 20,000,000.00 - [budget 80.00 %]
    22193/0000 - [analytical code name 02]: 350,000,000.00 - [budget 44.00 %]
    22194/0000 - [analytical code name 03]: 45,000,000.00 - [budget 60.00 %]
    22195/0000 - [analytical code name 04]: 255,000,000.00 - [budget 20.00 %]

    02. category 02
    22196/0000 - [analytical code name 05]: 206,000,000.00 - [budget 99.00 %]
    22197/0000 - [analytical code name 06]: 80,000,000.00 - [budget 12.00 %]
    22198/0000 - [analytical code name 07]: 99,000,000.00 - [budget 6.00 %]
    22199/0000 - [analytical code name 08]: 1,000,000.00 - [budget 35.00 %]

Only if the records that you selected relevant has both "Balance Sheet Item" and "P&L Item" both in them in the "Sub Category" column then this should be the upper level grouping of the records before the category grouping. Follow the given example,

    Balance Sheet Items
        01. category 01
        ...
        02. category 02
    
    P&L Items
        01. category 03
        ...
        02. category 04
        ...

This is unnecessary if the subcategory column only has one vlaue form Balanace Sheet Items and P&L Items or either it includes empty values (" ") which you can ignore.

Important - You SHOULD RESPOND UNTIL THE USER ASKED RESULT IS COMPLETED. Give at least 10000 words of response or until every detail user has asked is completely answered.
    
The ground level report of the account balance of the company for october 2024 in xlsx format is given for you.

You do not have to be emotional or natural language in your response, you should be very precise and technical in your response as you are communicating with another AI.
"""

template_conversation_model = """
You are a conversational AI assistant who works with structered data of a powerplant company and report insights in natural language to the users, who will be the management level of the company.
I will share the relevant data that I got filtered from a ground level report of the powerplant company only regarding the month october of 2024 which was generated by a middle thiking AI agent [You should never expose about this thinking model to the user], and you will follow ALL of the rules below:

1/ You should always be conversational and friendly as you are handling the end user who is the management level of the company directly. But do not greet unnecessarily in the start go staright to the point.

2/ You should make a proper structure of the response and make it very easy to read and understand for the user. Make sure to explain if something is complex or technical in a very simple way.

3/ The given analytical code number which is the number that starts each record is a unique dimension that you need to return. Always start with that number in that value and then next the name and other details.
The categeories which you may find grouped are the topics that you should return in response and list down the records under of the each record. You need to do this only if the categories or topics are mentioned in the given related data that you are.

4/ You will be provided the domain knowledge of the powerplant company, so always make sure to blend this domain knwoledge with the data you are provided ONLY if necessary. The user's message can either be regarding the data on the report of the given period or either about the company and domain knowledge itself. If the user's message is completely abou this the domain/company answer this from refering to the brief given.

5/ If you get the response "IRRELEVANT" from the middle AI agent that means the middle agent has decided that the user's message is completely irrelevant to the given report or the data.
In this case explain to the user that the message is irrelevant and you are not able to provide any insights or information regarding this message as you only have access to the financial data of the october 2024 and the compared data of the previous month and the previous year to that period but the user's message also can be on the company or the domain knowledge if so that becomes an exception.
If the user's message is also irrelevant to any of this two and nothing related to the topic, then just simply tell the user that you are unable to help with this and refer him other cloud based, popular AI tools which can help him with this if that is relevant for the user's message. If the calculation is already done and givne to you by the middle agent then you simply have to present this to the user nothing else.

6/ If you are giving a summary of something, like a total amount of a certain category that user specify, give a total amount or the sum that user has asked first but in this case give some key break down of which sub categories you used WITH NUMBERS related to the each category here secondly. giving numbers in this breakdown is really important if there is any.
If this happenes also try to give the percentage in number next to the actual number as well. But if this breakdown goes very long in context give atleast 5 to 6 points and say etc in natural language.

7/ Give the results in sub topics (categories) using the "Category" column when the user's message is asking for a list of things. Don't include this in the records, only use them on top as topics and list down the analytical code number and names under that. You only need to do this when the user asks list of things can be grouped using the Category column.
There is no need to mention about the previous month figure unless the user asks for it. Always mention the budget variance percentage in square brakcets as shown in the sample response below.
Here is a sample format of the output you should return;

01. category 01
22192/0000 - analytical code name 01: 20,000,000.00 - budget 80.00 %
22193/0000 - analytical code name 02: 350,000,000.00 - budget 44.00 % 
22194/0000 - analytical code name 03: 45,000,000.00 - budget 60.00 %
22195/0000 - analytical code name 04: 255,000,000.00 - budget 20.00 %

02. category 02
22196/0000 - analytical code name 05: 206,000,000.00 - budget 99.00 %
22197/0000 - analytical code name 06: 80,000,000.00 - budget 12.00 %
22198/0000 - analytical code name 07: 99,000,000.00 - budget 6.00 %
22199/0000 - analytical code name 08: 1,000,000.00 - budget 35.00 %

*notice that the analytical code name and budget variance percentage is not in square brakcets you should not return them in square brakcets just give it in the sample format only unless the user asks for a different format, when given the relevant data it has only given as that for the ease of identifying, and th subtopic numbers in brakcets are also unncessary in the final result.

Only if the records that you selected relevant has both "Balance Sheet Item" and "P&L Item" both in them in the "Sub Category" column then this should be the upper level grouping of the records before the category grouping. Follow the given example,

    Balance Sheet Items
        01. category 01
        ...
        02. category 02
    
    P&L Items
        01. category 03
        ...
        02. category 04
        ...

This is unnecessary if the subcategory column only has one vlaue form Balanace Sheet Items and P&L Items or either it includes empty values (" ") which you can ignore.

This was the response given by the middle AI agent to going through the ground level data of the company. You should use the totals and calculations provided in this reponse for the final reponse. This also includes relevant data from the ground level report. This was generated by a middle thinking AI agent [You should never expose about this thinking model to the user]:
{middle_agent_response}

*This may include csv file path or download link or advise to check the generated csv. You should always ignore these lines as they won't be passed on to the user.

Here is the brief of the powerplant company and the domain knowledge of the company. Do not give text straight from this just only understand this an explain as an conversational assistant ; {acwa_company_brief}
"""

template_conversation_model_w_data = """
You are a conversational AI assistant who works with structered data of a powerplant company and report insights in natural language to the users, who will be the management level of the company.
I will share the relevant data that I got filtered from a ground level report of the powerplant company only regarding the month october of 2024 which was generated by a middle thiking AI agent [You should never expose about this thinking model to the user], and you will follow ALL of the rules below:

1/ You should always be conversational and friendly as you are handling the end user who is the management level of the company directly. But do not greet unnecessarily in the start go staright to the point.

2/ You SHOULD RESPOND UNTIL THE USER ASKED RESULT IS COMPLETED. Give at least 10000 words of response or until every detail user has asked is completely answered.

3/ The given analytical code number which is the number that starts each record is a unique dimension that you need to return. Always start with that number in that value and then next the name and other details.
The categeories which you may find grouped are the topics that you should return in response and list down the records under of the each record. You need to do this only if the categories or topics are mentioned in the given related data that you are.

4/ If you get the response "IRRELEVANT" from the middle AI agent that means the middle agent has decided that the user's message is completely irrelevant to the given report or the data.
In this case explain to the user that the message is irrelevant and you are not able to provide any insights or information regarding this message as you only have access to the financial data of the october 2024 and the compared data of the previous month and the previous year to that period but the user's message also can be on the company or the domain knowledge if so that becomes an exception.
If the user's message is also irrelevant to any of this two and nothing related to the topic, then just simply tell the user that you are unable to help with this with the data you have acces to.

5/ If you are giving a summary of something, like a total amount of a certain category that user specify, give a total amount or the sum that user has asked first but in this case give some key break down of which sub categories you used WITH NUMBERS related to the each category here secondly. giving numbers in this breakdown is really important if there is any.
If this happenes also try to give the percentage in number next to the actual number as well. But if this breakdown goes very long in context give atleast 5 to 6 points and say etc in natural language. If the calculation is already done and givne to you by the middle agent then you simply have to present this to the user nothing else.

6/ Give the results in sub topics (categories) using the "Category" column when the user's message is asking for a list of things. Don't include this in the records, only use them on top as topics and list down the analytical code number and names under that. You only need to do this when the user asks list of things can be grouped using the Category column.
There is no need to mention about the previous month figure unless the user asks for it. Always mention the budget variance percentage in square brakcets as shown in the sample response below.
Here is a sample format of the output you should return;

01. category 01
\n22192/0000 - analytical code name 01: 20,000,000.00 - budget 80.00 %
\n22193/0000 - analytical code name 02: 350,000,000.00 - budget 44.00 % 
\n22194/0000 - analytical code name 03: 45,000,000.00 - budget 60.00 %
\n22195/0000 - analytical code name 04: 255,000,000.00 - budget 20.00 %

\n\n02. category 02
\n22196/0000 - analytical code name 05: 206,000,000.00 - budget 99.00 %
\n22197/0000 - analytical code name 06: 80,000,000.00 - budget 12.00 %
\n22198/0000 - analytical code name 07: 99,000,000.00 - budget 6.00 %
\n22199/0000 - analytical code name 08: 1,000,000.00 - budget 35.00 %

*notice that the analytical code name and budget variance percentage is not in square brakcets you should not return them in square brakcets just give it in the sample format only unless the user asks for a different format, when given the relevant data it has only given as that for the ease of identifying, and th subtopic numbers in brakcets are also unncessary in the final result.

Only if the records that you selected relevant has both "Balance Sheet Item" and "P&L Item" both in them in the "Sub Category" column then this should be the upper level grouping of the records before the category grouping. Follow the given example,

    Balance Sheet Items
        \n01. category 01
        \n...
        \n02. category 02
    
    \n\nP&L Items
        \n01. category 03
        \n...
        \n02. category 04
        \n...

This is unnecessary if the subcategory column only has one vlaue form Balanace Sheet Items and P&L Items or either it includes empty values (" ") which you can ignore.

This was the response given by the middle AI agent to going through the ground level data of the company. You should use the totals and calculations provided in this reponse for the final reponse. This also includes relevant data from the ground level report. This was generated by a middle thinking AI agent [You should never expose about this thinking model to the user]:
{middle_agent_response}

*This may include csv file path or download link or advise to check the generated csv. You should always ignore these lines as they won't be passed on to the user.

Here is the generated related data. Only use this data to brief down the response as the format given in the rule 07. Do not do any calculations or other analysis on this for that ALWAYS refer to the middle agent's reponse that's it;
{output_file_string}
"""

# The fucntion to filloiut the templates with values

def fill_template(template: str, context: dict) -> str:
    return template.format(**context)

# 2. Generate the model responses

def generate_data_assistant_response(message):

    # Upload a file with an "assistants" purpose
    file = client.files.create(
        file = open("gl-report-oct-24.xlsx", "rb"),
        purpose='assistants'
    )

    # 1️⃣ Create an assistant with the code interpreter tool
    assistant = client.beta.assistants.create(
        name="Data Assistant",
        model="gpt-4o-mini-2024-07-18",
        instructions=prompt_da,
        tools=[{"type": "code_interpreter"}],
        tool_resources={
            "code_interpreter":{
                "file_ids": [file.id]
            }
        }
    )

    # 2️⃣ Create a new thread for the conversation
    thread = client.beta.threads.create()

    # 4️⃣ Add user message to the thread
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=message
    )

    # 5️⃣ Create a run to process the conversation with the assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    # 6️⃣ Poll the run until it completes
    while run.status != "completed":
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

    # 7️⃣ Get all messages from the thread
    messages = client.beta.threads.messages.list(thread_id=thread.id)

    # Count input tokens
    if run.usage:
        total_tokens = run.usage.total_tokens
        input_tokens = run.usage.prompt_tokens
        percent_used = (total_tokens / 128000) * 100

    print(f"Input tokens: {input_tokens}")
    print(f"Total tokens used: {total_tokens} / 128000 ({percent_used:.2f}%)")

    response = "None returned"
    output_file_id = None

    print(messages.data)

    # 8️⃣ Print the assistant responses
    for msg in messages.data[::-1]:  # reverse the order
        if msg.role == "assistant":
            # Content can be a list of parts; extract text accordingly
            response = "".join(part.text.value for part in msg.content)
            print("\nAssistant:", response)
            if output_file_id is None:
                for attachment in msg.attachments:
                    output_file_id = attachment.file_id
        if msg.role == "user":
           # Content can be a list of parts; extract text accordingly
            text = "".join(part.text.value for part in msg.content)
            print("\nUser:", text)
        if msg.role == "system":
            # Content can be a list of parts; extract text accordingly
            print("\nSystem: system instructions")
    
    return response, output_file_id


def generate_response_for_thinking(message):
    with open("gl-report-24-october.txt", "r") as file:
            gl_report_oct_2024 = file.read()
    prompt_th = fill_template(template_thinking_model, {"gl_report_oct_2024": gl_report_oct_2024})
    response = client.chat.completions.create(
        model=thinking_model,
        messages=[
            {"role": "system", "content": prompt_th},
            {"role": "user", "content": message}
                ],
        tools=[{"type": "code_interpreter"}]
        )
    return response.choices[0].message.content

def generate_response_for_convo(message, middle_agent_response):
    with open("acwa_company_brief.txt", "r") as file:
        acwa_company_brief = file.read()
    prompt_co = fill_template(template_conversation_model, {"middle_agent_response":middle_agent_response, "acwa_company_brief": acwa_company_brief})
    # print(prompt_co)
    responseStream = client.chat.completions.create(
        model=conversational_model,
        messages=[
            {"role": "system", "content": prompt_co},
            {"role": "user", "content": message}
                ],
        stream=True
        )
    return responseStream

def generate_response_for_convo_w_data(message, middle_agent_response, output_file_string):
    prompt_co = fill_template(template_conversation_model_w_data, {"middle_agent_response":middle_agent_response, "output_file_string": output_file_string})
    # print(prompt_co)
    responseStream = client.chat.completions.create(
        model=conversational_model,
        messages=[
            {"role": "system", "content": prompt_co},
            {"role": "user", "content": message}
                ],
        max_tokens=4096,
        temperature=0.5,
        stream=True
        )
    return responseStream

def on_submit():
    st.session_state["show_loading"] = True
    st.session_state["show_result"] = False

def return_example(idx):
    examples = {
        1: "What is the total value of my long term loans for October 2024? give me a breakdown on this",
        2: "Can you provide a summary of staff expenditures for October 2024?",
        3: "Can you provide an analysis of the 'VAT INPUT' account based on the available information?",
        4: "Can you provide a breakdown of all the Current Liabilities we have for the given period of time?",
        5: "Give me a short progress overview about my company from the data you have access to."
    }
    st.session_state["message"] = examples.get(idx, "")
    st.session_state["show_result"] = False

def return_speech_text(speech_text):
        st.session_state["show_loading"] = True
        st.session_state["message"] = speech_text
        on_submit()

def display_table(csv_string: str):
    try:
        csv_io = StringIO(csv_string)
        df = pd.read_csv(csv_io)
        return df
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        print(f"Error reading CSV: {e}")
        return None

# 6. Streamlit App
def main():
    if "question_count" not in st.session_state:
        st.session_state["question_count"] = 0
    if "show_loading" not in st.session_state:
        st.session_state["show_loading"] = False
    if "show_result" not in st.session_state:
        st.session_state["show_result"] = False

    st.set_page_config(
        page_title="AI Finance Assistant :satellite::milky_way:", page_icon=":milky_way:")
    
    tab_chat, tab_summary, tab_balance, tab_income = st.tabs(["Assistant", "Summary", "Balance Data", "Income Data"])

    # st.header("ACWA Power :satellite::milky_way:")
    tab_chat.header("AI Finance Assistant :satellite::milky_way:")

    if "message" not in st.session_state:
        st.session_state["message"] = ""

    message = tab_chat.text_area("type", key="message", label_visibility="collapsed", height=150)

    # Add empty space before buttons
    tab_chat.write("")
    
    # # First row of buttons
    # col1, col2, col3 = tab_chat.columns([1, 1, 1])
    # with col1:
    #     tab_chat.button("Long Term Loans", key="btn1", on_click=lambda: (return_example(1), on_submit()), use_container_width=True)
    # with col2:
    #     tab_chat.button("Breakdown of Current Liabilities", key="btn4", on_click=lambda: (return_example(4), on_submit()), use_container_width=True)
    # with col3:
    #     tab_chat.button("Analysis on VAT INPUT Account", key="btn3", on_click=lambda: (return_example(3), on_submit()), use_container_width=True)
    

    # # Second row of buttons
    # col4, col5 = tab_chat.columns([1, 1])
    # with col4:
    #     tab_chat.button("Staff Spending Summary (Oct 2024)", key="btn2", on_click=lambda: (return_speech_text(""), return_example(2), on_submit()), use_container_width=True)
    # with col5:
    #     tab_chat.button("Brief Progress Overview of the Company", key="btn5", on_click=lambda: (return_example(5), on_submit()), use_container_width=True)

    # Enter button row
    tab_chat.button("Enter", key="submit", type="primary", on_click=on_submit, use_container_width=True)

    # speech_text = speech_to_text(language='en', use_container_width=True, just_once=True, key='STT')
    # if speech_text:
    #     pass
    #     return_speech_text(speech_text)
    # audio_value = st.audio_input("voice input",label_visibility="collapsed")

    # Add empty space after buttons
    tab_chat.write("")

    with tab_summary:
        components.iframe(os.getenv('SUMMARY_PBI_EMBED_URL'), width=1500, height=600, scrolling=True)
    with tab_balance:
        components.iframe(os.getenv('BALANCE_PBI_EMBED_URL'), width=1500, height=600, scrolling=True)
    with tab_income:
        components.iframe(os.getenv('INCOME_PBI_EMBED_URL'), width=1500, height=600, scrolling=True)
    
    if st.session_state["message"] and st.session_state["show_loading"]:
        message = st.session_state["message"]
        if message:
            print()
            print("-"*100)
            st.session_state["question_count"] += 1
            print(f"\nQEUSTION ID : Q{st.session_state['question_count']}")
            print("QUESTION :", message)
            # result_th = generate_response_for_thinking(message)]
            print("\nTHINKING... :")
            
            result_da, output_file_id = generate_data_assistant_response(message)
            output_file_string = ""

            result_co = ""
            expand_data_placeholder = tab_chat.empty()
            response_placeholder = tab_chat.empty()
            if output_file_id:
                output_file = client.files.content(output_file_id)
                output_file_bytes = output_file.read()
                output_file_string = output_file_bytes.decode('utf-8')
                output_table = display_table(output_file_string)
                if output_table is not None:
                    expand_data_placeholder.expander("See related records").write(output_table)
                else:
                    print("nothing returned displaying table")
                # with open("./generated-samples/related-data.csv", "wb") as file:
                #     file.write(output_file_bytes)
                # print("Output file downloaded as output.csv")

            try:
                result_co_stream = generate_response_for_convo_w_data(message, result_da, output_file_string)
            except Exception as e:
                st.error(e,"Input token exceeded error: The input is too large for the model. Please try a shorter or simpler query.")
                result_co_stream = generate_response_for_convo(message, result_da)
                return

            for chunk in result_co_stream:
                if chunk.choices[0].delta.content:
                    result_co += chunk.choices[0].delta.content
                    # print(result_co, end="", flush=True)
                    # Styled output inside the placeholder
                    response_placeholder.markdown(result_co)

            st.session_state["show_result"] = True
            st.session_state["show_loading"] = False

if __name__ == '__main__':
    main()