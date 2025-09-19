from PIL import Image
import streamlit as st
import pandas as pd
import json
import requests
from datetime import datetime
import os
import boto3
from botocore.exceptions import ClientError
import urllib.parse
import re
from data_processing import calucate_metrics
from utility import check_password
import concurrent.futures
import botocore.session
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
#must be first command to set the page config
st.set_page_config("PDT Data Cleaning", layout="wide", page_icon="icon.ico")

# Do not continue if valid_password is not True.
if not check_password():
   st.stop()

if 'post_rerun_message' in st.session_state:
    message_info = st.session_state.pop('post_rerun_message') # Use pop to get and remove
    if message_info["type"] == "success":
        st.success(message_info["text"])
    elif message_info["type"] == "error": # You can extend this for other message types
        st.error(message_info["text"])

# Load and display EDB logo
try:
    #logo = Image.open("edb_logo.png")
    #st.image(logo, width=150)
    print()
except FileNotFoundError:
    st.warning("edb_logo.png not found. Skipping logo display.")


# Title and description
st.title("Get Job Function, Level, Industry predictions from CSV File")

# Instructions
instructions = """

1. Upload a CSV file containing "Company Name" and "Job Title" columns.
2. Click the "Get Predictions" button for a file to see its predicted Function, Level, and Industry.
3. Edit the spreadsheet until everything is correct or download and file and edit it on excel.
4. Press submit if you edited the file on the website or upload the edited csv.

"""
st.header("Instructions")
st.text(instructions)
# Get region and strip any whitespace
region = os.environ.get('AWS_DEFAULT_REGION2')
s3 = boto3.client('s3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID2'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY2'),
    region_name=region
)
S3_METRICS_BUCKET = 'user-corrections' # Define bucket name
S3_METRICS_KEY = 'metrics/all_metrics.json' # Define the single JSON file key
REQUIRED_COLUMNS = ["Company Name", "Job Title"]
# Retrieve credentials safely
aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID2')
aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY2')






# Initialize session state
if 'uploaded_files_list' not in st.session_state:
    st.session_state.uploaded_files_list = []
if 'predictions_made_once_global' not in st.session_state: # Flag to indicate if any prediction has been made in the session
    st.session_state.predictions_made_once_global = False

# File uploader logic
# Only show the uploader if no predictions have been made globally yet
if not st.session_state.predictions_made_once_global:
    newly_uploaded_files = st.file_uploader(
        "Choose CSV file(s)", accept_multiple_files=False, key="main_file_uploader",type=["csv"],
    )

    if newly_uploaded_files:
        # Ensure uploaded_files_list is always a list of UploadedFile objects
        st.session_state.uploaded_files_list = [newly_uploaded_files]
        # When new files are uploaded, existing file-specific states for files
        # no longer in the list will persist but won't be accessed.
        # The loop below will re-initialize states for the new set of files if needed.
elif not st.session_state.uploaded_files_list and st.session_state.predictions_made_once_global:
    # This case handles when predictions were made, uploader disappeared,
    # and then for some reason the uploaded_files_list became empty (e.g., all files processed and cleared).
    # It allows the uploader to reappear if no files are listed and predictions were made.
    st.info("Predictions have been made. To upload new files, click the button below.")
    if st.button("Upload New Files Again"):
        st.session_state.predictions_made_once_global = False
        st.session_state.uploaded_files_list = []
        # Clear all file-specific states by iterating through session state keys if necessary,
        # or just rely on rerun re-evaluating from scratch for file processing.
        # For simplicity, the main loop will handle re-initialization for any new files.
        st.rerun()

function_labels = [
    "Accounting", "Administrative and Human Resources", "Community and Social Work",
    "Consulting, Strategy and Market Research", "Data Science or Analytics", "Design",
    "Education and Training", "Engineering", "Entrepreneurship", "Finance",
    "Healthcare Services", "Legal", "Marketing and Communications", "Operations",
    "Others", "Product Management", "Research and Innovation",
    "Sales and Business Development", "Sustainability", "Technology"
]
level_labels = [
    "Associate / Executive / Junior Specialist", "C-Suite",
    "EVP / SVP / VP / Director / Lead Specialist", "Manager / Senior Specialist"
]
industry_labels= [
    "Aerospace and Defense", "Arts, Entertainment and Hospitality", "Banking and Finance",
    "Built Environment", "Consumer and Retail", "Education", "Energy and Natural Resources",
    "Government and Non-Profit", "Healthcare", "Information and Communication Technology",
    "Life Sciences", "Manufacturing", "Others", "Professional Services", "Real Estate",
    "Social Services", "Sports"
]

def get_predictions(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts function, level, and industry for job entries in a DataFrame
    using a batch inference endpoint.

    Args:
        df_input (pd.DataFrame): Input DataFrame with 'Company Name' and 'Job Title' columns.

    Returns:
        pd.DataFrame: The DataFrame with 'Predicted Function', 'Predicted Level',
                      and 'Predicted Industry' columns added.
    """
    df = df_input.copy() # Work on a copy to avoid modifying the original DataFrame

    df["Company Name"] = df["Company Name"].astype(str)
    df["Job Title"] = df["Job Title"].astype(str)
    df['combined_text'] = df['Company Name'] + ' ' + df['Job Title']

    # Prepare the list of texts for batch inference
    texts_for_batch = df['combined_text'].tolist()

    # The URL where the Litserve app is running. Adjust if necessary.
    # Ensure this URL is correct for your Litserve deployment.
    #cpu inference endpoint for now
    JOB_FUNCTION_URL = "https://ezm27lqzj3akdhwxfiqd2qtwcy0bfcbq.lambda-url.ap-southeast-1.on.aws"
    JOB_LEVEL_URL = "https://m7abxso2no23asuhzex2mt46t40wbrjk.lambda-url.ap-southeast-1.on.aws"
    JOB_INDUSTRY_URL = "https://gn6vx4afizgwazczbnfbaz7ob40kuiqm.lambda-url.ap-southeast-1.on.aws"
    # Prepare the payload for batch inference
    payload = {"texts": texts_for_batch}

    predicted_functions = []
    predicted_levels = []
    predicted_industries = []

    try:
        # Make a single POST request for batch prediction to the /predict endpoint
        endpoints = [
        (f"{JOB_FUNCTION_URL}/predict", "function"),
        (f"{JOB_LEVEL_URL}/predict", "level"),
        (f"{JOB_INDUSTRY_URL}/predict", "industry"),
    ]
        def post_request(url):
            session = botocore.session.Session()
            session.set_credentials(
                access_key=os.environ.get('AWS_ACCESS_KEY_ID2'),
                secret_key=os.environ.get('AWS_SECRET_ACCESS_KEY2')
            )
            sigv4 = SigV4Auth(session.get_credentials(),
                                service_name="lambda",
                                region_name="ap-southeast-1")
            request = AWSRequest(method="POST", url=url, data=json.dumps(payload))
            request.headers['Content-Type'] = 'application/json'
            
            sigv4.add_auth(request)
            
            # Prepare the signed request
            signed = request.prepare()
            
            # Make the actual request
            resp = requests.post(signed.url, headers=signed.headers, data=signed.body)
            resp.raise_for_status()
            return resp.json()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(post_request, url): key for url, key in endpoints}
            results = {}
            for future in concurrent.futures.as_completed(futures):
                key = futures[future]
                results[key] = future.result()

        function_json = results["function"]
        level_json = results["level"]
        industry_json = results["industry"]
        # Process the batch response
        # The response structure for batch will have lists under "function", "level", "industry"
        # Each element in these lists corresponds to an input text in the order it was sent.

        for item in function_json:
            predicted_functions.append(item['predicted_label'])
        for item in level_json:
            predicted_levels.append(item['predicted_label'])
        for item in industry_json:
            predicted_industries.append(item['predicted_label'])
    except requests.exceptions.RequestException as e:
        print(f"Error during batch request: {e}")
        # Return the original DataFrame or an empty one if an error occurs
        # Have error handling to avoid breaking the app
        st.error(f"Failed to get predictions: {e}")
        st.warning("Please check your internet connection or the Litserve endpoint.")
        return None

    # Add the predicted columns to the DataFrame
    df['Predicted Function'] = predicted_functions
    df['Predicted Level'] = predicted_levels
    df['Predicted Industry'] = predicted_industries
    #limit the options so that users do not screw up
    df['Predicted Function'] = df['Predicted Function'].astype(pd.CategoricalDtype(function_labels))
    df['Predicted Level'] = df['Predicted Level'].astype(pd.CategoricalDtype(level_labels))
    df['Predicted Industry'] = df['Predicted Industry'].astype(pd.CategoricalDtype(industry_labels))
# Remove the temporary combined_text column
    df.drop(columns=['combined_text'], inplace=True)
    # Reorder the columns to have the predicted columns at the start
    cols = df.columns.tolist()
    predicted_cols = ['Predicted Function', 'Predicted Level', 'Predicted Industry']
    other_cols = [col for col in cols if col not in predicted_cols]
    df = df[predicted_cols + other_cols]
    return df
def create_hyperlink_for_col(df, col_name):
    '''
    Create a Google search URL for a given text in a specific column.
    Args:
        df (pd.DataFrame): The DataFrame containing the column.
        col_name (str): The name of the column for which the Google search URL is created.
        Returns: the DataFrame with Google search URLs in the specified column.
    '''
    df[col_name] = df[col_name].apply(
        lambda x: f"https://www.google.com/search?q={urllib.parse.quote_plus(str(x))}"
    )
    return df
def extract_original_text(html_string):
    # Check if the string is a Google search URL
    if isinstance(html_string, str) and html_string.startswith('https://www.google.com/search?q='):
        # Handle direct URL format - extract the query parameter and decode it
        match = re.search(r"https://www\.google\.com/search\?q=([^&]+)", html_string)
        if match:
            return urllib.parse.unquote_plus(match.group(1))
        else:
            return html_string
    else:
        # If it's not a Google search URL, return as is
        return html_string
def reverse_hyperlink_for_col(df, col_name):
    """
    Remove hyperlinks from a given column and extract the original text.
    Args:
        df (pd.DataFrame): The DataFrame containing the column with hyperlinks.
        col_name (str): The name of the column from which to remove hyperlinks.
    Returns:
        pd.DataFrame: The DataFrame with original text restored in the specified column.
    """
    
    df[col_name] = df[col_name].apply(extract_original_text)
    return df
def logging(edited_df, pushed_df,filename):
    """
    Log the metrics of the edited DataFrame compared to the pushed DataFrame.
    Args:
        edited_df (pd.DataFrame): The DataFrame after user edits.
        pushed_df (pd.DataFrame): The DataFrame that was pushed to the server.
    Returns:
        None
    """
    metrics = calucate_metrics(edited_df, pushed_df)
    #add the timestamp to the metrics and the file name
    metrics['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics['filename'] = filename
    return metrics
# Main processing loop for uploaded files
if st.session_state.uploaded_files_list:
    for uploaded_file in st.session_state.uploaded_files_list:
        # Use a more robust unique ID if file names can collide, e.g., uploaded_file.file_id
        # For now, using name, assuming they are distinct enough for keys.
        file_name_key_part = uploaded_file.name.replace(" ", "_").replace(".", "_") + "_" + str(uploaded_file.file_id)


        predictions_key = f"{file_name_key_part}_predictions_df"
        show_predictions_key = f"{file_name_key_part}_show_predictions"
        current_df_key = f"{file_name_key_part}_current_df"
        edited_df_key = f"{file_name_key_part}_edited_df"

        # Initialize file-specific session states
        st.session_state.setdefault(predictions_key, None)
        st.session_state.setdefault(show_predictions_key, False)
        st.session_state.setdefault(current_df_key, None) # Will be populated if file is valid
        st.session_state.setdefault(edited_df_key, None)

        try:
            # Only read and process the file if its current_df_key is not yet set or needs refresh
            # This check helps if the file was already processed in a previous run within the same list
            if st.session_state.get(current_df_key) is None or uploaded_file.file_id != st.session_state.get(f"{file_name_key_part}_processed_id"):

                uploaded_file.seek(0) 
                df_from_file = pd.read_csv(uploaded_file)
                st.session_state[current_df_key] = df_from_file
                st.session_state[f"{file_name_key_part}_processed_id"] = uploaded_file.file_id # Mark as processed

                missing_columns = [col for col in REQUIRED_COLUMNS if col not in df_from_file.columns]
                if missing_columns:
                    st.error(f"File '{uploaded_file.name}': Missing required columns: {', '.join(missing_columns)}")
                    st.session_state[current_df_key] = None # Invalidate
                else:
                    # Save the file (consider if this should be done only once)
                    save_dir = "user_input"
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, uploaded_file.name)
                    uploaded_file.seek(0)  
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    # st.success(f"File '{uploaded_file.name}' processed and input saved.")

        except pd.errors.EmptyDataError:
            st.error(f"Error: The file '{uploaded_file.name}' is empty.")
            st.session_state[current_df_key] = None
        except Exception as e:
            st.error(f"Error processing file '{uploaded_file.name}': {e}")
            st.session_state[current_df_key] = None

        # Display UI elements for this file if its DataFrame is loaded
        if st.session_state.get(current_df_key) is not None:
            #st.write(f"### Processing: {uploaded_file.name}") # Header for each file section
            # Display original or predicted data
            if st.session_state[show_predictions_key] and st.session_state.get(predictions_key) is not None:
                st.write(f"**Editable Predictions for {uploaded_file.name}:**")
                st.subheader("Edit your data ⬇️ or Download predictions to edit in excel")
                column_config = {
                    "Company Name": st.column_config.LinkColumn(
                        help="Click to Google search the company name",
                        display_text=r"https://www\.google\.com/search\?q=([^&]+)"   # Regex to extract text between > and </a>
                    ),
                    "Job Title": st.column_config.TextColumn(
                        "Job Title",
                        disabled=True
                    ),
                    "Predicted Function": st.column_config.SelectboxColumn(
                        "Predicted Function ✏️",
                        options=function_labels,
                        required=True,
                        help="Click to select or edit function"
                    ),
                    "Predicted Level": st.column_config.SelectboxColumn(
                        "Predicted Level ✏️",
                        options=level_labels,
                        required=True,
                        help="Click to select or edit level"
                    ),
                    "Predicted Industry": st.column_config.SelectboxColumn(
                        "Predicted Industry ✏️",
                        options=industry_labels,
                        required=True,

                        help="Click to select or edit industry"
                    )
                }
                
                edited_df = st.data_editor(st.session_state[predictions_key], key=f"editor_{file_name_key_part}", num_rows="dynamic", disabled=['Job Title'],column_config=column_config, use_container_width=True)
                st.session_state[edited_df_key] = edited_df

                col1, col2, col3 = st.columns(3) # Create columns for three buttons

                with col1:
                    #convert company name url to actual company name
                    csv_df_for_download = edited_df.copy()
                    csv_df_for_download = reverse_hyperlink_for_col(csv_df_for_download, 'Company Name')
                    
                    csv_data = csv_df_for_download.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"Download Edited CSV for {uploaded_file.name}",
                        data=csv_data,
                        file_name=f"predicted_edited_{uploaded_file.name}",
                        mime='text/csv',
                        key=f"download_{file_name_key_part}"
                    )
                
                with col2:
                    st.write("**Upload from Computer:**")
                    uploaded_from_drive = st.file_uploader(
                        f"Choose file to upload for {uploaded_file.name}",
                        type=['csv'],
                        key=f"drive_uploader_{file_name_key_part}"
                    )
                    
                    if uploaded_from_drive is not None:
                        if st.button(f"Save Selected File to Server", key=f"save_drive_file_{file_name_key_part}"):
                            try:
                                # Read the uploaded file from hard drive
                                drive_df = pd.read_csv(uploaded_from_drive)
                                
                                # Save to s3 bucket 
                                server_save_dir = "user_cleaned"
                                os.makedirs(server_save_dir, exist_ok=True)
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                base_name, extension = os.path.splitext(uploaded_from_drive.name)
                                server_file_name = f"from_drive_{base_name}_{timestamp}{extension}"
                                server_save_path = os.path.join(server_save_dir, server_file_name)
                                #we will save the file to the s3 bucket
                                drive_df = drive_df[REQUIRED_COLUMNS + ['Predicted Function', 'Predicted Level', 'Predicted Industry']]
                                # Save the DataFrame to the hard drive
                                drive_df.to_csv(server_save_path, index=False)
                                s3.upload_file(server_save_path, 'user-corrections', server_file_name)
                                #log the metrics
                                original_prediction_key = f"{file_name_key_part}_original_predictions"
                                current_metrics_entry = logging(st.session_state[original_prediction_key], drive_df, server_file_name)
                                
                                # Append metrics to a single JSON file in S3
                                try:
                                    response = s3.get_object(Bucket=S3_METRICS_BUCKET, Key=S3_METRICS_KEY)
                                    all_metrics_data = json.loads(response['Body'].read().decode('utf-8'))
                                    if not isinstance(all_metrics_data, list):
                                        # If existing data is not a list, start a new list with the current entry
                                        all_metrics_list = [current_metrics_entry]
                                    else:
                                        all_metrics_list = all_metrics_data
                                        all_metrics_list.append(current_metrics_entry)
                                except ClientError as e:
                                    if e.response['Error']['Code'] == 'NoSuchKey':
                                        # File doesn't exist, so this is the first entry
                                        all_metrics_list = [current_metrics_entry]
                                    else:
                                        # Some other S3 error
                                        st.error(f"Error reading existing metrics from S3: {e}")
                                        all_metrics_list = [current_metrics_entry] # Fallback to current metrics only
                                except json.JSONDecodeError:
                                    st.warning("Could not decode existing metrics JSON from S3. Starting with new metrics.")
                                    all_metrics_list = [current_metrics_entry] # Fallback

                                updated_metrics_json = json.dumps(all_metrics_list, indent=2) # indent for readability
                                s3.put_object(Body=updated_metrics_json, Bucket=S3_METRICS_BUCKET, Key=S3_METRICS_KEY)
                                os.remove(server_save_path)
                               # 1) capture the current password flag
                                pw_state = st.session_state.get("password_correct")

                                # 2) delete everything except the password flag
                                for k in list(st.session_state.keys()):
                                    if k != "password_correct":
                                        del st.session_state[k]

                                # 3) restore the password flag and set the post-rerun message
                                st.session_state["password_correct"] = pw_state
                                st.session_state["post_rerun_message"] = {
                                    "type": "success",
                                    "text": f"Successfully submitted '{server_file_name}' to S3."
                                }

                                # 4) reset any other globals you need
                                st.session_state.uploaded_files_list = []
                                st.session_state.predictions_made_once_global = False

                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Failed to save file: {e}")
                
                with col3:
                    #submission to server
                    csv_df_for_submit = edited_df.copy()
                    csv_df_for_submit = reverse_hyperlink_for_col(csv_df_for_submit, 'Company Name')
                    csv_data_submit = csv_df_for_submit.to_csv(index=False).encode('utf-8')
                    if st.download_button(f"Submit Final CSV for {uploaded_file.name}", key=f"submit_final_csv_{file_name_key_part}",data=csv_data_submit,
                        file_name=f"predicted_edited_{uploaded_file.name}",
                        mime='text/csv'):
                        if st.session_state.get(edited_df_key) is not None:
                            final_submission_dir = "user_cleaned"
                            os.makedirs(final_submission_dir, exist_ok=True)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            base_name, extension = os.path.splitext(uploaded_file.name)
                            submission_file_name = f"submitted_{base_name}_{timestamp}{extension}"
                            submission_save_path = os.path.join(final_submission_dir, submission_file_name)
                            # Upload to S3 bucket
                            #only upload the Company Name and Job Title columns with the predictions
                            pushed_df = st.session_state[edited_df_key][REQUIRED_COLUMNS + ['Predicted Function', 'Predicted Level', 'Predicted Industry']]
                            pushed_df = reverse_hyperlink_for_col(pushed_df, 'Company Name')
                            # Save the edited DataFrame to the final submission path
                            pushed_df.to_csv(submission_save_path, index=False)
                            s3.upload_file(submission_save_path, 'user-corrections', submission_file_name)
                            #log the metrics
                            original_prediction_key = f"{file_name_key_part}_original_predictions"
                            current_metrics_entry = logging(st.session_state[original_prediction_key], csv_data_submit, submission_file_name)

                            # Append metrics to a single JSON file in S3
                            try:
                                response = s3.get_object(Bucket=S3_METRICS_BUCKET, Key=S3_METRICS_KEY)
                                all_metrics_data = json.loads(response['Body'].read().decode('utf-8'))
                                if not isinstance(all_metrics_data, list):
                                    # If existing data is not a list, start a new list with the current entry
                                    all_metrics_list = [current_metrics_entry]
                                else:
                                    all_metrics_list = all_metrics_data
                                    all_metrics_list.append(current_metrics_entry)
                            except ClientError as e:
                                if e.response['Error']['Code'] == 'NoSuchKey':
                                    # File doesn't exist, so this is the first entry
                                    all_metrics_list = [current_metrics_entry]
                                else:
                                    # Some other S3 error
                                    st.error(f"Error reading existing metrics from S3: {e}")
                                    all_metrics_list = [current_metrics_entry] # Fallback to current metrics only
                            except json.JSONDecodeError:
                                st.warning("Could not decode existing metrics JSON from S3. Starting with new metrics.")
                                all_metrics_list = [current_metrics_entry] # Fallback
                            
                            updated_metrics_json = json.dumps(all_metrics_list, indent=2) # indent for readability
                            s3.put_object(Body=updated_metrics_json, Bucket=S3_METRICS_BUCKET, Key=S3_METRICS_KEY)
                            try:
                                #redundant when we did logging in the instance itself
                                st.session_state[edited_df_key].to_csv(submission_save_path, index=False)
                                #Delete the csv file
                                os.remove(submission_save_path)
                                # 1) capture the current password flag
                                pw_state = st.session_state.get("password_correct")

                                # 2) delete everything except the password flag
                                for k in list(st.session_state.keys()):
                                    if k != "password_correct":
                                        del st.session_state[k]

                                # 3) restore the password flag and set the post-rerun message
                                st.session_state["password_correct"] = pw_state
                                st.session_state["post_rerun_message"] = {
                                    "type": "success",
                                    "text": f"Successfully submitted '{submission_file_name}' to S3."
                                }

                                # 4) reset any other globals you need
                                st.session_state.uploaded_files_list = []
                                st.session_state.predictions_made_once_global = False

                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"Failed to submit '{submission_file_name}': {e}")
                        else:
                            st.warning("No edited data available to submit.")
            else: # Show original uploaded data or prompt for predictions
                st.write(f"**Uploaded Data for {uploaded_file.name}:**")
                st.dataframe(st.session_state[current_df_key])

            # "Get Predictions" button
            # Show button if predictions haven't been shown yet for this file
            if not st.session_state[show_predictions_key]:
                if st.button(f"Get Predictions for {uploaded_file.name}", key=f"predict_{file_name_key_part}"):
                    df_to_predict = st.session_state[current_df_key].copy()
                    with st.spinner(f"Generating predictions for {uploaded_file.name}..."):
                        #save the original prediction
                        st.session_state[predictions_key] = get_predictions(df_to_predict)
                        original_prediction = f"{file_name_key_part}_original_predictions"
                        st.session_state[original_prediction] = st.session_state[predictions_key].copy()
                        #we edit the predictions dataframe to add hyperlinks to the company name col
                        st.session_state[predictions_key] = create_hyperlink_for_col(st.session_state[predictions_key], 'Company Name')
        
                    if st.session_state[predictions_key] is None:
                        st.error(f"Failed to get predictions for {uploaded_file.name}. Please check the logs.")
                        st.rerun()
                    else:
                        st.session_state[show_predictions_key] = True
                        st.session_state.predictions_made_once_global = True # Update global flag
                        st.rerun()
