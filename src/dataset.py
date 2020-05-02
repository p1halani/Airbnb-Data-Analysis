import pandas as pd
from dateutil import parser
from geopy.distance import vincenty
from src.categorical import DataPreprocess

def no_smoking(house_rules_txt):
    is_no_smoking = int('no smoking' in str(house_rules_txt).lower() or 'smoking is not allowed' in str(house_rules_txt).lower())
    return is_no_smoking

def no_pets(house_rules_txt):
    is_no_pets = int('no pets' in str(house_rules_txt).lower() or 'no smoking or pets' in str(house_rules_txt).lower())
    return is_no_pets

def rate_to_fraction(rate_str):
    if rate_str == 0:
        rate_frac = None
    else:
        rate_frac = 0.01*int(str(rate_str).split('%')[0])

        return rate_frac

def prices_to_numbers(price_string):
    price_numeric = float(str(price_string).replace(',', '').split('$')[-1])
    return price_numeric

if __name__ == '__main__':
    seattle_listings = pd.read_csv('input/listings_seattle.csv')
    price_cols = ['price', 
              'weekly_price', 
              'monthly_price', 
              'security_deposit', 
              'cleaning_fee', 
              'extra_people']
    for col in price_cols:
        seattle_listings[col] = seattle_listings[col].apply(prices_to_numbers)
    
    seattle_listings.dropna(axis=0, how='any', subset=['host_since'], inplace=True)
    seattle_listings['host_experience'] = seattle_listings.last_scraped.apply(lambda s: parser.parse(s)) - seattle_listings.host_since.apply(lambda s: parser.parse(s))
    seattle_listings.host_experience = seattle_listings.host_experience.astype(str).apply(lambda s: int(s.split(' ')[0]))
    seattle_listings.drop(['last_scraped', 'host_since'], axis=1, inplace=True)

    seattle_listings['house_rules_no_smoking'] = seattle_listings.house_rules.apply(no_smoking)
    seattle_listings['house_rules_no_pets'] = seattle_listings.house_rules.apply(no_pets)
    seattle_listings.drop(['house_rules'], axis=1, inplace=True)

    seattle_listings['host_verification_level'] = seattle_listings.host_verifications.apply(lambda s: len(s.split(', ')))

    seattle_listings['host_response_fraction'] = seattle_listings.host_response_rate.fillna(0).apply(rate_to_fraction)
    seattle_listings['host_acceptance_fraction'] = seattle_listings.host_acceptance_rate.fillna(0).apply(rate_to_fraction)
    seattle_listings.drop(['host_acceptance_rate'], axis=1, inplace=True)
    seattle_listings.drop(['host_response_rate'], axis=1, inplace=True)

    room_type_dict = {'Shared room': 1, 'Private room': 2, 'Entire home/apt': 3, 'Hotel room': 4}
    seattle_listings.room_type.replace(room_type_dict, inplace=True)

    seattle_listings['is_TV'] = seattle_listings.amenities.apply(lambda s: int('TV' in str(s)[1:].split(',')))
    seattle_listings['is_Wifi'] = seattle_listings.amenities.apply(lambda s: int('Wifi' in str(s)[1:].split(',')))
    seattle_listings['amenities_number'] = seattle_listings.amenities.apply(lambda s: len(str(s)[1:].split(',')))
    seattle_listings.drop('amenities', axis=1, inplace=True)

    seattle_listings.dropna(axis=0, how='any', subset=['bathrooms', 'beds', 'bedrooms'], inplace=True)

    seattle_listings_1 = seattle_listings.T.drop_duplicates().T
    duplicate_cols = [i for i in seattle_listings.columns if i not in seattle_listings_1.columns]
    del seattle_listings_1
    seattle_listings.drop(duplicate_cols, axis=1, inplace=True)
    seattle_description = seattle_listings.describe().T

    seattle_description['iqr_max'] = seattle_description['75%'] + 1.5 * (seattle_description['75%'] - seattle_description['25%'])
    seattle_description['iqr_min'] = seattle_description['75%'] - 1.5 * (seattle_description['75%'] - seattle_description['25%'])
    cols_to_drop_outliers = ['maximum_nights', 'minimum_nights', 'price']

    for col in cols_to_drop_outliers:
        seattle_listings = seattle_listings[((seattle_listings[col] > seattle_description.loc[col, 'iqr_min']) & (seattle_listings[col] < seattle_description.loc[col, 'iqr_max']))]

    lat_center_dict = {1: 47.6062100, 2: 42.3584300, 3: 55.675940}
    lon_center_dict = {1: -122.3320700, 2: -71.0597700, 3: 12.5655300}
    seattle_listings = pd.concat([seattle_listings], keys = [1])
    seattle_listings.reset_index(level=0, inplace=True)
    seattle_listings.rename(columns={'level_0': 'city_index'}, inplace=True)
    seattle_listings['lat_center'] = seattle_listings.city_index.replace(lat_center_dict)
    seattle_listings['lon_center'] = seattle_listings.city_index.replace(lon_center_dict)
    seattle_listings['distance_center'] = seattle_listings.apply(lambda x: vincenty((x['latitude'], x['longitude']), (x['lat_center'], x['lon_center'])).km, axis = 1)

    drop_columns = ['thumbnail_url','license','id','listing_url',
                    'scrape_id','picture_url','host_url','host_thumbnail_url','host_picture_url',
                    'name','summary','space','description','neighborhood_overview','notes','transit',
                    'access','interaction','host_name','host_location','host_about','host_neighbourhood',
                    'street','neighbourhood','state','zipcode','market','smart_location','city',
                    'country_code','country','calendar_updated','calendar_last_scraped','first_review',
                    'last_review','jurisdiction_names','experiences_offered','property_type','bed_type',
                    'cancellation_policy','square_feet','monthly_price','weekly_price','host_verifications',
                    'neighbourhood_group_cleansed','neighbourhood_cleansed','lat_center','lon_center',
                    'longitude','latitude','city_index','maximum_nights','minimum_maximum_nights',
                    'availability_90','host_verification_level','minimum_nights_avg_ntm',
                    'maximum_nights_avg_ntm','minimum_minimum_nights','calculated_host_listings_count_shared_rooms',
                    'review_scores_cleanliness','is_TV','review_scores_location','host_response_time',
                    'house_rules_no_pets','availability_60','maximum_minimum_nights','house_rules_no_smoking',
                    'availability_30','minimum_nights','host_is_superhost','is_location_exact',
                    'instant_bookable','maximum_maximum_nights','review_scores_communication',
                    'review_scores_checkin','require_guest_profile_picture','host_identity_verified',
                    'review_scores_accuracy','require_guest_phone_verification','is_Wifi',
                    'host_has_profile_pic','is_business_travel_ready','requires_license','has_availability'
                    ]

    cols = []

    cat_feats = DataPreprocess( seattle_listings,
                                drop_columns=drop_columns,
                                categorical_features=cols, 
                                encoding_type="label",
                                handle_na=True
                            )

    data_transformed = cat_feats.fit_transform()

    data_transformed.to_hdf('input/preprocessed_seattle_listings.h5', key='seattle_listings', mode='w')

    seattle_reviews = pd.read_csv('input/reviews_seattle.csv')
    seattle_reviews.dropna(subset=['comments'], how='any', axis=0, inplace=True)
    vader_polarity_compound = lambda s: (SentimentIntensityAnalyzer().polarity_scores(s))['compound']
    seattle_reviews['polarity'] = seattle_reviews.comments.map(vader_polarity_compound)

    seattle_reviews.to_hdf('input/preprocessed_seattle_reviews.h5', key='seattle_reviews', mode='w')