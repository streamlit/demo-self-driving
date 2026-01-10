[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/streamlit/demo-self-driving)

# Snowflake Streamlit Hands OnDemo Setup:

1. Create Conda environment  
   1. [How to connect Streamlit to Snowflake](https://www.youtube.com/watch?v=SgWxkAdjK78)   
   2. Conda create \-n snowpark python=3.11  
      1. Note: Python 3.12 is not compatible with Snowpark yet\!

2. Activate Conda environment and install 2 packages:  
   1. Conda activate snowpark  
   2. Pip install snowflake-snowpark-python streamlit

3. Clone an existing Streamlit Repo or create your own.   
   1. Original Repo: [https://github.com/streamlit/demo-self-driving](https://github.com/streamlit/demo-self-driving)  
   2. Fork in my repo with small additions to make it run on the most recent version of Streamlit (1.40.1) [https://github.com/sfc-gh-mjohnson/streamlit-demo-self-driving/](https://github.com/sfc-gh-mjohnson/streamlit-demo-self-driving/)   
   3. Branch with more major additions to incorporate a 4th tab in the navigation, and a Snowflake Editable Data Frame.  [https://github.com/sfc-gh-mjohnson/streamlit-demo-self-driving/tree/snowflake\_connection](https://github.com/sfc-gh-mjohnson/streamlit-demo-self-driving/tree/snowflake_connection) 

   

4. Navigate to the top-level folder in your app directory.  
   1. streamlit run streamlit\_app.py

5. After you have verified that you can get Streamlit working and load the first 3 tabs of the app, then you can start forming your Snowflake connection.   
     
6. Add .streamlit/secrets.toml to git local environment  
   1. Create new file  
   2. Create sample file so you know the syntax  
   3. Add to .gitignore so it’s not synced up and visible in my public github\!

7. Set up credentials to Snowflake in a config.toml file [https://docs.streamlit.io/develop/tutorials/databases/snowflake](https://docs.streamlit.io/develop/tutorials/databases/snowflake) 

   

   \[connections.snowflake\]

   account     \= "sfpscogs-mj\_aws"

   user        \= "NORQUEST\_SVC"

   password    \= "-----"

   role        \= "SYSADMIN"

   warehouse   \= "TUTORIAL\_WH"

   database    \= "TUTORIAL\_DB"

   schema      \= "NORQUEST"

   

8. Deploy the app on Streamlit Community  
   1. Sign in to Streamlit.IO and create an account  
   2. Connect to your Github Repo  
   3. Deploy the App [https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)   
   4. Choose an App URL  
   5. Click Advanced Settings \- Specify Python 3.11.   
      1. If you don’t specify this before you deploy the app, most of the app will work, but the Snowflake connection will be broken. You will have to delete the app and recreate it.   
   6. If you are ready, insert your credentials from your Secrets.TOML file here.

9. Confirm Snowflake connection  
   1. Go to the 4th tab “Give Feedback” and you should see a form and a data frame that is pulling data from the Snowflake Account that you connected.  

10. Streamlit Apps  
    1. [https://streamlit-cloud-example-apps-streamlit-app-sw3u0r.streamlit.app/](https://streamlit-cloud-example-apps-streamlit-app-sw3u0r.streamlit.app/)   
       1. List of all apps that are good inspirations  
    2. [https://streamlit-example-app-bug-report-streamlit-app-lrm3fx.streamlit.app/](https://streamlit-example-app-bug-report-streamlit-app-lrm3fx.streamlit.app/)  
       1. Github: [example-app-bug-report](https://github.com/streamlit/example-app-bug-report/blob/main/streamlit_app.py)   
    3. [https://streamlit-demo-uber-nyc-pickups-streamlit-app-456wus.streamlit.app/](https://streamlit-demo-uber-nyc-pickups-streamlit-app-456wus.streamlit.app/)  
       1. Github [demo-uber-nyc-pickups](https://github.com/streamlit/demo-uber-nyc-pickups)  
    4. [https://streamlit-demo-self-driving-streamlit-app-8jya0g.streamlit.app/](https://streamlit-demo-self-driving-streamlit-app-8jya0g.streamlit.app/)  
       1. Github [demo-self-driving](https://github.com/streamlit/demo-self-driving)  





# Streamlit Demo: The Udacity Self-driving Car Image Browser

This project demonstrates the [Udacity self-driving-car dataset](https://github.com/udacity/self-driving-car) and [YOLO object detection](https://pjreddie.com/darknet/yolo) into an interactive [Streamlit](https://streamlit.io) app.

The complete demo is [implemented in less than 300 lines of Python](https://github.com/streamlit/demo-self-driving/blob/master/streamlit_app.py) and illustrates all the major building blocks of Streamlit.

![Making-of Animation](https://raw.githubusercontent.com/streamlit/demo-self-driving/master/av_final_optimized.gif "Making-of Animation")

## How to run this demo
```
pip install --upgrade streamlit opencv-python
streamlit run https://raw.githubusercontent.com/streamlit/demo-self-driving/master/streamlit_app.py
```

### Questions? Comments?

Please ask in the [Streamlit community](https://discuss.streamlit.io).


