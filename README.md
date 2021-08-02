# An Exploration of NFT Artwork
For this project, I built a Streamlit web application that contains an NFT artwork generator and an NFT artwork recommendation engine. The artwork generator was made by creating an autoencoder, and the artwork recommendation engine uses euclidean distances between an input image and a database of NFT artworks that I curated. 

Data: 
I scraped over 500,000 tweets from Twitter (via Twint) using keywords related to non-fungible tokens and stored the data in a Mongo database. Within those tweets there were about 100,000 photos, which I downloaded and stored in a separate database. Most of those photos were NFT artworks, which allowed me to create the artwork generator and the recommendation engine. See the presentation slides for the final results! 
