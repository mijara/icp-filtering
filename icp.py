from openai import OpenAI
from dotenv import load_dotenv
import os 
import json 
import pandas as pd 
from datetime import datetime
import logging
from collections import defaultdict

logging.basicConfig(
    filename='logs/gpt4_usage.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def log_gpt4_response(response):
    content = response.choices[0].message.content
    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens
    log_data = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "content": content
    }
    logging.info(json.dumps(log_data, ensure_ascii=False))

def format_data(client,data, fields=None):
    fields = {
        "About": "What does the company do?",
        "Mission": "What problems do they solve for?",
        "Products": "List of products mentioned",
        "Pricing": "List of pricing tiers",
        "Customers": "Information about customers, including names and industries",
        "Testimonials": "Customer testimonials, including the name of the person providing the testimonial and their role if mentioned.",
        "Industries & Segments": "Industries and market segments that the company focuses on"
    }


    field_instructions = "\n".join([f"{field}: {description}" for field, description in fields.items()])

    system_message = f"""You are an intelligent web data extraction agent. Your task is to perform extractive question answering on the input website text corpus and extract the following fields: {fields.keys()} into JSON format. The JSON should strictly contain only the relevant information extracted from the input website text, 
    with no additional commentary, explanations, or extraneous information. In case you can't find the answer to any of the required fields, answer as null.
    Please process the following text and provide the output in pure JSON format with no words before or after the JSON.
    Extract the following information with these descriptions:
    {field_instructions}"""


    user_message = f"Page content:\n\n{data}\n\nInformation to extract: {fields}"



    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        response_format={ "type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    )

    if response and response.choices:
        log_gpt4_response(response)
        formatted_data = response.choices[0].message.content.strip()
        print(f"Formatted data received from API: {formatted_data}")

        try:
            parsed_json = json.loads(formatted_data)
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            print(f"Formatted data that caused the error: {formatted_data}")
            raise ValueError("The formatted data could not be decoded into JSON.")
        
        return parsed_json
    else:
        raise ValueError("The OpenAI API response did not contain the expected choices data.")
    

def synthesize_context(client,context_list):
    combined_input = "\n\n".join(str(obj) for obj in context_list)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Synthesise the following JSON object attribute-wise without missing any attribute:\n\n{combined_input}\n\nOutput the consolidated attributes."}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        response_format={ "type": "json_object"},
        messages=messages
    )
    log_gpt4_response(response)
    synthesized_context = response.choices[0].message.content.strip()
    return synthesized_context
    

def generate_icps(client,consolidated_context_object):
    icp_prompt = f"""
    Given the following information about a company, please identify the ideal customer profiles (ICPs) for the company's products. The company offers a range of products and services described in the provided contextualization. The goal is to derive profiles that would most benefit from the company's ability to solve specific challenges and enhance certain operations, allowing businesses to focus on their core objectives.

    Consider the following when identifying the ICPs:
    1. The specific industries and segments that would most benefit from the company's products (e.g., those mentioned in the contextualization).
    2. The pain points these potential customers are facing that the company's products can address (e.g., difficulties, costs, or inefficiencies highlighted in the contextualization).
    3. The size and type of companies that would be ideal customers (e.g., small to medium-sized enterprises, large corporations, startups in relevant industries).
    4. The role of the decision-makers within these companies who would be most interested in the company's offerings (e.g., Chief Marketing Officer, Head of Digital Marketing, Product Manager, as applicable).

    Using this contextualization, generate 8 detailed ICPs that SDRs can use for prospecting the company's products.

    Output the ICPs in a structured JSON format with the following attributes: 
    "Industry/Segment", "Pain Points", "Size/Type", "Decision-Makers".

    Contextualization Object: {consolidated_context_object}
    """
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    icp_messages = [
        {"role": "system", "content": "You are an expert in sales and marketing."},
        {"role": "user", "content": icp_prompt}
    ]

    icp_response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        response_format={ "type": "json_object"},
        messages=icp_messages
    )
    log_gpt4_response(icp_response)
    icps = icp_response.choices[0].message.content.strip()
    return icps

def safe_extract(data, key, fallback="N/A"):
    return data.get(key, fallback) if data.get(key) else fallback

# Function to generate the refined prompt for GPT-4o
def generate_icp_filtering_prompt(icps, lead_context, linkedin_data, company_context):
    # Safely extract LinkedIn-related data
    job_title = linkedin_data['job_title']
    headline = linkedin_data['headline']
    summary = linkedin_data['summary']
    company_name = linkedin_data['company_name']
    company_industry = linkedin_data['company_industry']

    # Safely extract Phyllo's context
    phyllo_about = safe_extract(company_context, 'About', 'N/A')
    phyllo_mission = safe_extract(company_context, 'Mission', 'N/A')
    phyllo_products = safe_extract(company_context, 'Products', 'N/A')
    phyllo_pricing = safe_extract(company_context, 'Pricing', 'N/A')
    phyllo_customers = safe_extract(company_context, 'Customers', 'N/A')
    phyllo_testimonials = safe_extract(company_context, 'Testimonials', 'N/A')
    phyllo_industries_segments = safe_extract(company_context, 'Industries & Segments', 'N/A')

    prompt = f"""
You are an expert B2B sales analyst specializing in lead filtering for a company. The company, described below, provides solutions that require alignment with specific Ideal Customer Profiles (ICPs).

### Company Context (Phyllo):
- **About:** {phyllo_about}
- **Mission:** {phyllo_mission}
- **Products:** {phyllo_products}
- **Pricing:** {phyllo_pricing}
- **Customers:** {phyllo_customers}
- **Testimonials:** {phyllo_testimonials}
- **Industries & Segments:** {phyllo_industries_segments}

Your task is to assess a lead’s fit against the provided ICPs and the company's context.

### Inputs:
1. **ICPs:** 
    - A list of Ideal Customer Profiles, each containing:
        - **Industry/Segment**
        - **Pain Points**
        - **Company Size/Type**
        - **Decision-Makers**

2. **Lead Context Object:**
    - **About:** {safe_extract(lead_context, 'About')}
    - **Mission:** {safe_extract(lead_context, 'Mission')}
    - **Products:** {safe_extract(lead_context, 'Products')}
    - **Pricing:** {safe_extract(lead_context, 'Pricing')}
    - **Customers:** {safe_extract(lead_context, 'Customers')}
    - **Testimonials:** {safe_extract(lead_context, 'Testimonials')}
    - **Industries & Segments:** {safe_extract(lead_context, 'Industries & Segments')}

3. **LinkedIn Profile Information:**
    - **Job Title:** {job_title}
    - **Headline:** {headline}
    - **Summary:** {summary}
    - **Company Name:** {company_name}
    - **Company Industry:** {company_industry}

### Task:
1. **Industry/Segment Match:** 
   - Assess whether the lead's industry or segment directly relates to Phyllo’s target industries and segments. Assign a "High Match," "Partial Match," or "No Match" accordingly.

2. **Pain Points Evaluation:** 
   - Identify if the lead's pain points are directly related to challenges that can be addressed by Phyllo’s products and services. Be specific about which pain points are aligned and how.

3. **Company Size/Type Compatibility:** 
   - Evaluate whether the lead’s company size and type match the specifications in the ICPs and Phyllo's ideal customer context (e.g., small agencies scaling operations, large platforms requiring data integration).

4. **Decision-Maker Role Assessment:** 
   - Compare the roles mentioned in the lead’s context (e.g., from testimonials or inferred from company structure) with the decision-makers listed in the ICPs. Use the LinkedIn profile title and summary to assess alignment. Provide a judgment on alignment (e.g., "Exact Match," "Similar Role," "Different Role").

5. **Overall Fit Scoring and Categorization:** 
   - Based on the evaluations, assign a detailed score (e.g., 1-5) for each criterion (Industry/Segment, Pain Points, Size/Type, Decision-Makers). Calculate an overall fit score and categorize the lead as "High Fit," "Moderate Fit," or "Low Fit."

### Output:
Provide a detailed analysis of the lead’s fit, including:
- **Matching ICP(s):** Specify which ICP(s) the lead aligns with and the level of alignment.
- **Fit Scores:** A detailed score for each criterion and an overall fit score.
- **Categorization:** Clear categorization of the lead as "High Fit," "Moderate Fit," or "Low Fit."
- **Rationale:** Provide a concise explanation for each score and the overall categorization.
- **Recommended Next Steps:** Suggest specific actions based on the categorization (e.g., "Proceed to outreach," "Conduct further research," "Deprioritize").
    """

    return prompt

# Phyllo context
company_context = {
  "About": "Phyllo is a comprehensive API platform that provides access to creator-consented data across multiple social media platforms such as YouTube, Instagram, TikTok, Twitch, and more. It serves as a universal API gateway enabling businesses and developers to access authenticated influencer and creator data for enhancing influencer marketing strategies, social media integration, and the creator economy.",
  "Mission": "Phyllo aims to empower brands and marketers by simplifying and improving the collection and utilization of creator data, solving the challenge of accessing unified, verified creator data for developing effective influencer and marketing strategies. By providing easy and secure access to creator-consented data, Phyllo facilitates data democratization in the creator economy and Web3, allowing businesses and developers to focus on building their products.",
  "Products": [
    "Phyllo Universal API",
    "Phyllo Connect SDK",
    "Social Listening API",
    "Twitch Chat API",
    "Instagram APIs",
    "Influencer Marketing Tools",
    "Engagement API",
    "Identity API",
    "Publish APIs",
    "Phyllo Social Media Demographics API",
    "Phyllo Income API",
    "Data Platform",
    "Bubble Plug-ins",
    "Creator Linkage",
    "Audience API",
    "LinkedIn Creator Search API",
    "insightIQ"
  ],
  "Pricing": [
    "Free Plan",
    "Free Access",
    "Paid Subscriptions",
    "Enterprise Packages",
    "Customizable subscription based on brand size, current influencer marketing status, and goals",
    "Three subscription tiers: Growth, Scale, and Enterprise"
  ],
  "Customers": "Working with companies in the creator economy space, including Beacons, Creative Juice, Creator.co, Karat, MagicLinks, Bintango, Nerve, as well as others like BintanGo, 456 Growth, and Impulze.",
  "Testimonials": [
    {
      "text": "Phyllo has been an invaluable partner for us, allowing seamless access to comprehensive creator data.",
      "name": "Rahul Bansal",
      "role": "Co-founder & CTO, BintanGo"
    },
    {
      "text": "Phyllo has transformed our experience of manual data collection from influencers at 456 Growth.",
      "name": None,
      "role": None
    },
    {
      "text": "Phyllo has helped Impulze immensely in scaling our product.",
      "name": None,
      "role": None
    },
    {
      "text": "Phyllo helped us build faster and easier by abstracting away various social media platform developer APIs.",
      "name": "Vinod Verma",
      "role": "Cofounder, Creator.Co"
    },
    {
      "text": "Creator Platform integration is an important part of our infrastructure and we have offloaded all of this to Phyllo.",
      "name": None,
      "role": None
    },
    {
      "text": "We initially integrated with Instagram and YouTube APIs directly, but quickly realized that the time for each integration was not feasible; further, the effort to maintain each integration was exorbitant. Now that we are able to offload all of that to Phyllo, we're able to focus on our core algorithms and platform without worrying about the infrastructure.",
      "name": "Vinod Verma",
      "role": "Cofounder, Creator.Co"
    }
  ],
  "Industries & Segments": [
    "Influencer Marketing",
    "Social Media Platforms",
    "Data Integration",
    "Social Media Integration",
    "Creator Economy",
    "Video Streaming",
    "Campaign Analytics",
    "Fintech",
    "Web3",
    "Financial Services",
    "Ecommerce",
    "Fashion",
    "Beauty",
    "Retail",
    "B2B Influencer Marketing",
    "Creator Tools",
    "Social Identity Verification",
    "Digital Marketplaces",
    "Community Platforms",
    "Design and Content Creation Tools",
    "Social Media Management",
    "Data Analysis and Analytics"
  ]
}


icps = [
    {
        "Industry/Segment": "Influencer Marketing, Campaign Analytics",
        "Pain Points": "Difficulty in accessing unified, verified creator data to create data-driven marketing strategies.",
        "Size/Type": "Small to medium-sized agencies focused on scaling their operations by leveraging comprehensive influencer data.",
        "Decision-Makers": ["Chief Marketing Officers", "Head of Digital Marketing", "Campaign Strategists"]
    },
    {
        "Industry/Segment": "Social Media Integration, Social Media Platforms",
        "Pain Points": "Struggling with manual integration of APIs from multiple social media platforms, leading to inefficiencies.",
        "Size/Type": "Medium to large platforms seeking to enhance their offerings by integrating comprehensive data analytics.",
        "Decision-Makers": ["Product Managers", "CTOs", "Head of Partnerships"]
    },
    {
        "Industry/Segment": "Video Streaming, Community Platforms",
        "Pain Points": "Need for seamless chat and engagement features alongside creator insights to enhance viewer experience.",
        "Size/Type": "Large corporations or emerging platforms wanting to improve viewer engagement and content management.",
        "Decision-Makers": ["Chief Operations Officers", "Head of Content Strategy"]
    },
    {
        "Industry/Segment": "Fintech, Financial Services",
        "Pain Points": "Requirement for verifying social media identities and income data of influencers for financial services offerings.",
        "Size/Type": "Startups or scale-ups developing financial products for the creator economy.",
        "Decision-Makers": ["CEOs", "Head of Product Development", "Chief Data Officers"]
    },
    {
        "Industry/Segment": "Ecommerce, Retail",
        "Pain Points": "Challenges in identifying and working with relevant influencers to boost brand visibility and sales.",
        "Size/Type": "Brands focusing on expanding their influencer marketing efforts for growth.",
        "Decision-Makers": ["Head of Ecommerce", "Chief Marketing Officers", "Brand Managers"]
    },
    {
        "Industry/Segment": "Design and Content Creation Tools, Creator Tools",
        "Pain Points": "Need for integrating creator data to provide personalized tools and insights for creators.",
        "Size/Type": "Medium-sized companies seeking to improve tool efficiency and customer experience.",
        "Decision-Makers": ["Head of Product Design", "CTO", "Marketing Directors"]
    },
    {
        "Industry/Segment": "Web3, Data Integration",
        "Pain Points": "Necessity for democratized access to creator data to build Web3 applications and services.",
        "Size/Type": "Companies innovating in the Web3 space, integrating creator and community data solutions.",
        "Decision-Makers": ["Founders", "Product Leads", "Head of Innovation"]
    },
    {
        "Industry/Segment": "Digital Marketplaces, Social Identity Verification",
        "Pain Points": "Ensuring authenticity and credibility of user profiles by utilizing comprehensive public data insights.",
        "Size/Type": "Large operational marketplaces aiming to bolster trust and user engagement.",
        "Decision-Makers": ["CEOs", "Head of User Experience", "Directors of Trust & Safety"]
    }
]


def filter_lead_with_gpt(client,prompt):
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a B2B sales expert with a deep understanding of SaaS products."},
            {"role": "user", "content": prompt}
        ]
    )
    log_gpt4_response(response)
    return response.choices[0].message.content.strip()

def main():
    load_dotenv()
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    lead_website_contents_df=pd.read_excel("/Users/jidin/icp-filtering/lead_website_contents.xlsx")
    # responses=defaultdict(list)
    # for index, row in lead_website_contents_df.iterrows():
    #     website=row["website"]
    #     content=row["content"]
    #     length=100000
    #     for i in range(0,len(content),length):
    #         responses[website].append(format_data(client=client,data=content[i:i+length]))
    # print("Extractive QA done")
    # file_path = "responses_output.json"
    # with open(file_path, 'w') as file:
    #     json.dump(responses, file, indent=4)
    # lead_contexts={}
    # for website in responses:
    #     lead_contexts[website]=synthesize_context(client,responses[website])
    # file_path = "output.json"
    # with open(file_path, 'w') as file:
    #     json.dump(lead_contexts, file, indent=4)
    # lead_contexts_df = pd.DataFrame(lead_contexts.items(), columns=['Lead', 'Context'])
    # lead_contexts_df.to_excel('lead_contexts.xlsx', index=False)
    # print("Synthesizing context and writing to file done. ")
    lead_contexts_df=pd.read_excel("/Users/jidin/icp-filtering/lead_contexts.xlsx")
    merged_df = pd.concat([lead_website_contents_df, lead_contexts_df], axis=1)
    merged_df['ICP Scoring'] = None
    for index, row in merged_df.iterrows():
        context_value = row["Context"]
        if context_value is not None and not pd.isna(context_value):
            lead_context = json.loads(context_value)
        else:
            lead_context = {}
        linkedin_data = {
            "job_title": row.get("Title", "N/A"),
            "headline": row.get("headline", "N/A"),
            "summary": row.get("summary", "N/A"),
            "company_name": row.get("company_name", "N/A"),
            "company_industry": row.get("company_industry", "N/A")
        }
        prompt = generate_icp_filtering_prompt(icps, lead_context, linkedin_data, company_context)
        result = filter_lead_with_gpt(client=client,prompt=prompt)
        merged_df.at[index, 'ICP Scoring'] = result
    merged_df.to_excel('merged_df.xlsx', index=False)



if __name__ == "__main__":
    main()
