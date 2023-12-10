# CS_598_Project_4-Movie_Recommender_System

# How are general recommendations handled? (Non-IBCF)

We will recommend movies with the highest score.

To define what we mean by highest scoring:

Using purely the normalized rating obviously favors movies that have a low number of ratings, so we instead introduce a scoring function that places some percentage of the score weight on the number of ratings. We place more weight on the normalized rating (90%), with the last 10% weight is put on the number of ratings mainly to ensure that movies with one rating don't commonly end up at the top. This will prevent one or two people rating a movie a 5 causing it to be scored better than all time classics that are scored well on large volumes of votes, indicating they are more likely to be enjoyed as a blind recommendation. 

Finally, the importance of the number of ratings is winsorized beyond 1000 ratings. This corresponds to the logic that after 1000 ratings, more ratings are unlikely to signficantly change our identified score, so they just serve to unfairly weight popular movies.

This metric will be referred to as "score."

# Edge Case considerations while using IBCF

### User does not provide enough/any ratings to system 2 

- If 0 movies are rated: pad the list with the highest score movies (general, not from a particular genre) (Not needed as app doesn't take 0 input)

- If >0 but <10 movies are rated: use IBCF to recommend as many movies as possible, then pad this list with highest score movies 

### Movies provided by system 2 are low rated 

- Generally non-issue, as more popular movies will naturally bubble to the top

- Also not worth super curating, as we could take a “one man’s trash is another mans treasure” approach 

- Ultimately, cull recommending movies whose normalized rating is below 3.0, as this is universally seen as a below average movie

### No similar movies (or under some threshold of similarity) 

- Default to popular movies (defined by score) 
