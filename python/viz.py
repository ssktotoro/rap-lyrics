if __name__== '__main__':

    from python.non_json_scraper import SongScraper
    import pandas as pd
    from python.utils import preprocess_lyrics
    from python.utils import apply_lda
    import numpy as np
    from sklearn.externals import joblib

    # scraper = SongScraper()
    # drake_songs = scraper.scrape_rap_genius('drake')
    # drake_album_list_by_song = [song.album for song in drake_songs]
    # drake_title_list_by_song = [song.title for song in drake_songs]
    # drake_lyrics_list_by_song = [song.lyrics for song in drake_songs]
    # drake_songs_df = pd.DataFrame({'album': drake_album_list_by_song, 'title': drake_title_list_by_song,
    #                                'lyrics': drake_lyrics_list_by_song})
    # kanye_songs = scraper.scrape_rap_genius('kanye')
    # kanye_album_list_by_song = [song.album for song in kanye_songs]
    # kanye_title_list_by_song = [song.title for song in kanye_songs]
    # kanye_lyrics_list_by_song = [song.lyrics for song in kanye_songs]
    # kanye_songs_df = pd.DataFrame({'album': kanye_album_list_by_song, 'title': kanye_title_list_by_song,
    #                                'lyrics': kanye_lyrics_list_by_song})

    kanye_songs_df = joblib.load('kanye_songs_df.pkl')
    kanye_p_lyrics = preprocess_lyrics(kanye_songs_df['lyrics'])
    kanye_topic_prob, kanye_lda = apply_lda(kanye_p_lyrics, 10)
    kanye_p_lyrics = pd.DataFrame(np.array(kanye_p_lyrics))
    topic_evaluation_df = pd.merge(kanye_p_lyrics, kanye_topic_prob, left_index=True, right_index=True)
    topic_evaluation_df.to_csv('topic_Evaluation.csv', quoting=1, encoding='utf-8')
    kanye_lda_data = pd.DataFrame()
    kanye_lda_data['topic'] = kanye_lda[0]
    kanye_lda_data['probability for topic'] = kanye_lda[1]
    kanye_lda_data['lyrics'] = pd.DataFrame(np.array(kanye_p_lyrics))