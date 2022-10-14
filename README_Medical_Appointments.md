#  Wrangling Data from Twitter API and Other Sources
## by Banji R. Owabumoye
---

## Dataset
Datasets were gathered manually and programmatically (using python programming language codes) from three different sources. A twitter-archive-enhanced dataset stored as comma separated value (csv) file was manually downloaded from Udacity platform. The second dataset gathered is a tab separated value (tsv) file, labelled ‘image_predictions.tsv’. This dataset was programmatically downloaded from a Uniform Resource Locator (URL) through a library known as Requests. The third dataset was gathered from Twitter API using Tweepy package to query the Javascript Object Notation (JSON) file labelled ‘tweet_json.txt’. 

## Summary of Findings
Just like data gathering, the three datasets were assessed visually or manually and assessed programmatically. The datasets were visually assessed by looking at the dataframe of each dataset for quality and structural issues. Also, functions and methods were adopted to assess the datasets programmatically. This assessment methods revealed 10 quality issues and 4 structural issues. The

**Quality issues**
> - Doggo, floofer, pupper, and puppo columns have values as 'None'
> - Platform sources embedded in html code in source column
> - Timestamp datatype is object instead of datetime
> - Over 2000 null values in eachcolumn(in_reply_to_status_id, in_reply_to_user_id, retweeted_status_id, retweeted_status_user_id,    retweeted_status_timestamp)
> - Tweet_id datatype is int64 instead of object
> - Inconsistency in p1 column case, some are upper while some are lower and underscore in between words
> - Incosistency in p2 column case, some are uppr while some are lower and underscore in between words
> - Incosistency in p3 column case, some are uppr while some are lower and underscore in between words
> - Tweet_id datatype is int64 rather than object
> - Column names except tweet_id are not descriptive
> - Doggo, floofer, pupper, and puppo columns have values as 'None'
> - Platform sources embedded in html code in source column
> - Timestamp datatype is object instead of datetime

**Tidiness issues**
> - Doggo, floofer, pupper, and puppo should not have separate columns
> - Day_name, month, and year_month columns from timestamp column
> - Tweeter_df, image_df, and tweeter_api_df dataframes should merged together since they all have tweet_id column
    doggo, pupper, floofer, and puppo columns are no more needed

All the quality issues and structural issues are cleaned programmatically with python libraries (Pandas and Numpy). After testing and ascertaining with codes that the issues are corrected, all the three datasets are merged and stored as master dataset to a csv file named `twitter_archive_master dataset`.

## Key Insights for Presentation

* There is positive relationship between tweets retweeted and tweets that are favourites.
* WeRateDogs fans are more active between Monday through the midweek and some of them would be off the page during weekend.
* The data timeline was November 2015 to August 2017, which is 21 months. Within this period, the first five months (November 2015 – March 2016) recorded highest number of tweets with highest been in the month of December 2015. However, number of tweets dropped sharply from April 2016 and maintained the low trend till July 2017 which is 15 months period until the lowest number of tweets were recorded in August 2017.
* Among the four dog stages (pupper, doggo, puppo, and floofer) pupper recorded the highest tweet from Monday through Sunday.
* Among the four dog stages, the most popular which is also the youngest stage is pupper with 64% tweets.
* The tweet platforms used are iphone, vine, twitter, and tweetdeck. Iphone recorded 94%