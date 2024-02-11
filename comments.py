import pandas as pd
from googleapiclient.discovery import build
from bs4 import BeautifulSoup
import re

def clean_comment(comment):
    soup = BeautifulSoup(comment, 'html.parser')
    # Remove timestamp links
    comment_text = re.sub(r'<a.*?>.*?</a>', '', str(soup))
    #remove any html tags
    comment_text = re.sub(r'<.*?>', '', comment_text)
    return comment_text



# Set your API key or OAuth 2.0 credentials
API_KEY = "AIzaSyBItWlOlaAB-bHGaBdg2Dh5XBa5uAt9WfE"  # Replace with your API key

# Create a YouTube API client

youtube = build("youtube", "v3", developerKey=API_KEY)
def get_all_comments(video_id, n):
    comments_data = []
    next_page_token = None
    num = 0
    while num < n:
        # Request comments for the video
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            order = 'relevance',
            maxResults=100,  # Adjust as needed
            pageToken=next_page_token
        ).execute()

        # Extract comments and relevant information from the response
        for item in response.get("items", []):
            comment_data = [
                #get the id of the commenter
                item["snippet"]["topLevelComment"]["snippet"]["authorChannelId"]["value"],
                item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]]
                # item["snippet"]["topLevelComment"]["snippet"]["likeCount"]]
                # item["snippet"]["totalReplyCount"]]
            comments_data.append(comment_data)
            num += 1

        # Check if there are more pages of comments
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    
    # Create a DataFrame from the comments data
    df = pd.DataFrame(comments_data, columns=['Author','Comment'])
    df['cleaned_comment'] = df['Comment'].apply(clean_comment)
    return df


#-------------------------------------------------------------------
import sys
sys.path.append('..')
from LandingPage.main import analyse
from googleapiclient.discovery import build
import pandas as pd

# Setting YouTube API key
API_KEY = "AIzaSyBItWlOlaAB-bHGaBdg2Dh5XBa5uAt9WfE"

# Create a YouTube API client
youtube = build("youtube", "v3", developerKey=API_KEY)

def get_channel_id(channel_name):
    request = youtube.search().list(
        part="id",
        q=channel_name,
        type="channel"
    )
    response = request.execute()
    return response["items"][0]["id"]["channelId"] if response["items"] else None

def get_video_id(video_name):
    request = youtube.search().list(
        part="id",
        q=video_name,
        type="video"
    )
    response = request.execute()
    return response["items"][0]["id"]["videoId"] if response["items"] else None

def get_all_video_urls(channel_id):
    video_urls = []
    next_page_token = None
    nums = 0
    while nums < 9:
        request = youtube.search().list(
            part="id",
            channelId=channel_id,
            maxResults=50,  # Adjust as needed
            order="date",
            pageToken=next_page_token,
            type="video"
        )
        response = request.execute()

        for item in response.get("items", []):
            if nums > 9:
                break
            video_urls.append(item['id']['videoId'])
            nums += 1

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return video_urls


def comment_master(channel_name):
    # Get the channel ID
    channel_id = get_channel_id(channel_name)
    master_df = pd.DataFrame(columns = ['Author','cleaned_comment', 'negative'])
    if channel_id:
        # Get all video URLs from the channel
        video_urls = get_all_video_urls(channel_id)

        # Print or use video URLs as needed
        for i,url in enumerate(video_urls):
            try:
                df = get_all_comments(url,100)
                print(f'video {i} started')
                df['negative'] = df['cleaned_comment'].apply(analyse)
                filtered_authors = df[df['negative'] > 0.6]
                # print(filtered_authors.head())
                # print(filtered_authors.columns)
                master_df = pd.concat([master_df, filtered_authors], ignore_index=True)
                # print('Master df updated')
                # print(master_df.head())
                # df.to_csv(f'comments_{i}.csv', index=False)
            except:
                print(f'video {i} failed')
                pass
        master_df.to_csv(f'{channel_name}_comments.csv', index=False)
        grouped = master_df.groupby('Author')['negative'].nunique()
        authors_with_multiple_negatives = grouped[grouped > 1].index.tolist()
        if len(authors_with_multiple_negatives) > 0:
            # print('Troll found')
            # print(authors_with_multiple_negatives)
            for author in authors_with_multiple_negatives:
                print(master_df[master_df['Author'] == author])
            # print('Most negative comments across all videos are: ')
            # #sorted by negative score
            return master_df.sort_values('negative', ascending=False).head().to_dict(), authors_with_multiple_negatives

        else:
            # print('No troll found')
            # print('Most negative comments across all videos are: ')
            #sorted by negative score
            return master_df.sort_values('negative', ascending=False).head()



if __name__ == "__main__":
    channel_name = input('Enter the channel name: ')
    comment_master(channel_name)
            

