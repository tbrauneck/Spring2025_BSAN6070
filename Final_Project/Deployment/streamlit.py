import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# ✅ MUST come before any Streamlit command
st.set_page_config(page_title="Income Prediction App", layout="wide")

# custom_transformers.py
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class LOOImputer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, sigma=0.0):
        self.cols = cols
        self.sigma = sigma
        self.global_mean = None
        self.target_means = {}

    def fit(self, X, y):
        self.global_mean = np.mean(y)
        for col in self.cols:
            self.target_means[col] = {}
            categories = X[col].unique()
            for cat in categories:
                mask = X[col] == cat
                if mask.sum() > 1:
                    self.target_means[col][cat] = (y[mask].sum() - y[mask].iloc[0]) / (mask.sum() - 1)
                else:
                    self.target_means[col][cat] = self.global_mean
        return self

    def transform(self, X):
        X_transformed = X.copy()
        rng = np.random.default_rng()
        for col in self.cols:
            means = self.target_means[col]
            X_transformed[col] = X[col].apply(
                lambda x: means.get(x, self.global_mean) + rng.normal(0, self.sigma)
            )
        return X_transformed
# custom_transformers.py
from sklearn.base import BaseEstimator, TransformerMixin

class EnsureNumeric(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(pd.to_numeric, errors='coerce')

# Example of manual mappings for selected features
state_names = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "District of Columbia", "Florida", "Georgia",
    "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
    "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota",
    "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island",
    "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
    "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"
]
sex_map = {'Male': 0, 'Female': 1}
REGION_map = {
        'East South Central Div.': 1,
        'Pacific Division': 2,
        'Mountain Division': 3,
        'West South Central Div.': 4, 
        'New England Division': 5,
        'South Atlantic Division': 6, 
        'East North Central Div.': 7,
        'West North Central Div.': 8,
        'Middle Atlantic Division': 9
    }
MARST_map = {
        'Never married/single': 1,
        'Married, spouse present': 2, 
        'Divorced' : 3,
        'Separated': 4,
        'Widowed': 5,
        'Married': 6,
        'spouse absent': 7
    }
RACE_map = {
        'Black/African American': 1, 
        'White': 2, 
        'Two major races': 3, 
        'Other race, nec': 4,
        'Other Asian or Pacific Islander': 5, 
        'American Indian or Alaska Native': 6,
        'Chinese': 7, 
        'Three or more major races': 8,
        'Japanese': 9
    }
BPL_map = {
    'Georgia': 1, 'Alabama': 2, 'Florida': 3, 'Missouri': 4, 'Mexico': 5, 'New York': 6, 'California': 7, 'New Jersey': 8, 'North Carolina': 9, 'Nevada': 10,
    'India': 11, 'Michigan': 12, 'Maryland': 13, 'Pennsylvania': 14, 'Germany': 15, 'Indiana': 16, 'Illinois': 17, 'Colorado': 18, 'Tennessee': 19, 'Mississippi': 20,
    'Cuba': 21, 'Kentucky': 22, 'China': 23, 'West Indies': 24, 'Texas': 25, 'Central America': 26, 'New Hampshire': 27, 'SOUTH AMERICA': 28, 'Ohio': 29, 'Wisconsin': 30,
    'Oklahoma': 31, 'Washington': 32, 'Belgium': 33, 'Nebraska': 34, 'Maine': 35, 'Other USSR/Russia': 36, 'Iraq': 37, 'Massachusetts': 38, 'Virginia': 39, 'Alaska': 40,
    'Rhode Island': 41, 'Pacific Islands': 42, 'Louisiana': 43, 'South Carolina': 44, 'West Virginia': 45, 'Arizona': 46, 'Korea': 47, 'France': 48, 'Idaho': 49, 'Connecticut': 50,
    'Canada': 51, 'Lebanon': 52, 'District of Columbia': 53, 'Kansas': 54, 'Vietnam': 55, 'Puerto Rico': 56, 'Minnesota': 57, 'Oregon': 58, 'Guam': 59, 'Hawaii': 60,
    'Australia and New Zealand': 61, 'Delaware': 62, 'Arkansas': 63, 'Wyoming': 64, 'United Kingdom, ns': 65, 'Japan': 66, 'Philippines': 67, 'Iowa': 68, 'South Dakota': 69,
    'Italy': 70, 'New Mexico': 71, 'Nepal': 72, 'AFRICA': 73, 'Montana': 74, 'Utah': 75, 'North Dakota': 76, 'Laos': 77, 'Vermont': 78, 'Spain': 79, 'Malaysia': 80,
    'England': 81, 'Syria': 82, 'Greece': 83, 'Indonesia': 84, 'Turkey': 85, 'Thailand': 86, 'Yugoslavia': 87, 'Romania': 88, 'Europe, ns': 89, 'Poland': 90, 'Austria': 91,
    'Iran': 92, 'Netherlands': 93, 'Ireland': 94, 'Cambodia (Kampuchea)': 95, 'Saudi Arabia': 96, 'Singapore': 97, 'Hungary': 98, 'Portugal': 99, 'United Arab Emirates': 100,
    'Asia, nec/ns': 101, 'Kuwait': 102, 'Other n.e.c.': 103, 'Iceland': 104, 'Afghanistan': 105, 'Switzerland': 106, 'U.S. Virgin Islands': 107, 'Israel/Palestine': 108,
    'Atlantic Islands': 109, 'Scotland': 110, 'Lithuania': 111, 'Bulgaria': 112, 'Finland': 113, 'Albania': 114, 'Czechoslovakia': 115, 'Americas, n.s.': 116, 'Jordan': 117,
    'Sweden': 118, 'Denmark': 119, 'Latvia': 120, 'Yemen Arab Republic (North)': 121, 'American Samoa': 122, 'Norway': 123
    }
ANCESTR1_map = {
    'African-American': 1, 'Not Reported': 2, 'White/Caucasian': 3, 'Italian': 4, 'Mexican': 5, 'German': 6, 'Afro-American': 7, 'English': 8, 'Irish, various subheads,': 9,
    'United States': 10, 'European, nec': 11, 'Welsh': 12, 'French': 13, 'Spanish': 14, 'Asian Indian': 15, 'Scotch Irish': 16, 'Norwegian': 17, 'Uncodable': 18, 'Spaniard': 19,
    'Polish': 20, 'Chinese': 21, 'American Indian  (all tribes)': 22, 'Other Pacific': 23, 'French Canadian': 24, 'Scottish': 25, 'African': 26, 'Colombian': 27, 'Southern European, nec': 28,
    'Hungarian': 29, 'Swedish': 30, 'British': 31, 'Hispanic': 32, 'Ukrainian': 33, 'Kurdish': 34, 'Peruvian': 35, 'Puerto Rican': 36, 'Korean': 37, 'Slovak': 38, 'Greek': 39,
    'Mixture': 40, 'Austrian': 41, 'Scandinavian, Nordic': 42, 'Cuban': 43, 'Lebanese': 44, 'Swiss': 45, 'Vietnamese': 46, 'Dutch': 47, 'Chilean': 48, 'Australian': 49,
    'Belgian': 50, 'Russian': 51, 'Northern European, nec': 52, 'Hawaiian': 53, 'Brazilian': 54, 'Taiwanese': 55, 'Japanese': 56, 'Ecuadorian': 57, 'Yugoslavian': 58, 'Canadian': 59,
    'Guatemalan': 60, 'Honduran': 61, 'Western European, nec': 62, 'Salvadoran': 63, 'Pakistani': 64, 'Venezuelan': 65, 'Nuevo Mexicano': 66, 'Filipino': 67, 'Asian': 68, 'Danish': 69,
    'Jamaican': 70, 'Portuguese': 71, 'Panamanian': 72, 'Estonian': 73, 'Palestinian': 74, 'Mexican American': 75, 'Bulgarian': 76, 'Iranian': 77, 'Romanian': 78, 'Haitian': 79,
    'Egyptian': 80, 'Nepali': 81, 'Latvian': 82, 'Czechoslovakian': 83, 'Samoan': 84, 'Serbian': 85, 'Acadian': 86, 'Algerian': 87, 'Nigerian': 88, 'Eastern European, nec': 89,
    'Other Subsaharan Africa': 90, 'Jordanian': 91, 'Cameroonian': 92, 'Polynesian': 93, 'Syrian': 94, 'Latin American': 95, 'Slav': 96, 'Chamorro Islander': 97, 'Dominican': 98,
    'West Indian': 99, 'Armenian': 100, 'North American': 101, 'Trinidadian/Tobagonian': 102, 'Guamanian': 103, 'Lithuanian': 104, 'Indonesian': 105, 'Pacific Islander': 106,
    'Eskimo': 107, 'Slovene': 108, 'Other': 109, 'Finnish': 110, 'Zimbabwean': 111, 'New Zealander': 112, 'Mongolian': 113, 'Sicilian': 114, 'Laotian': 115, 'British Isles': 116,
    'Togo': 117, 'Assyrian/Chaldean/Syriac': 118, 'Liberian': 119, 'Iraqi': 120, 'Croatian': 121, 'Guyanese/British Guiana': 122, 'Other Asian': 123, 'Arab': 124, 'Basque': 125,
    'Marshall Islander': 126, 'South American': 127, 'Bengali': 128, 'Cambodian': 129, 'Malaysian': 130, 'Ghanian': 131, 'Congolese': 132, 'Ethiopian': 133, 'Argentinean': 134,
    'Tongan': 135, 'Kenyan': 136, 'Israeli': 137, 'Chicano/Chicana': 138, 'Burmese': 139, 'Hong Kong': 140, 'Thai': 141, 'Bohemian': 142, 'Belorussian': 143, 'Prussian': 144,
    'Icelander': 145, 'Afghan': 146, 'Eritrean': 147, 'West African': 148, 'Sri Lankan': 149, 'Macedonian': 150, 'Turkish': 151, 'South African': 152, 'Nicaraguan': 153,
    'Middle Eastern': 154, 'Germans from Russia': 155, 'Somalian': 156, 'Bolivian': 157, 'Sierra Leonean': 158, 'Spanish American': 159, 'Maltese': 160, 'Luxemburger': 161,
    'Moroccan': 162, 'Cantonese': 163, 'Bahamian': 164, 'Senegalese': 165, 'Albanian': 166, 'Other Arab': 167, 'Dutch West Indies': 168, 'Micronesian': 169, 'Hmong': 170,
    'Costa Rican': 171, 'Fijian': 172, 'Saudi Arabian': 173, 'Punjabi': 174, 'Moldavian': 175, 'Sudanese': 176, 'South American Indian': 177, 'Uruguayan': 178, 'Belizean': 179,
    'Central American Indian': 180, 'Flemish': 181, 'Georgian': 182, 'St Lucia Islander': 183, 'Yemeni': 184, 'Cape Verdean': 185, 'Barbadian': 186, 'Paraguayan': 187,
    'Central European, nec': 188, 'Libyan': 189, 'Okinawan': 190, 'Texas': 191, 'Other West Indian': 192, 'Tibetan': 193, 'Ugandan': 194, 'Cossack': 195, 'Uzbek': 196,
    'North African': 197, 'Rom': 198, 'British West Indian': 199, 'British Virgin Islander': 200, 'Anguilla Islander': 201, 'Grenadian': 202, 'Alsatian, Alsace-Lorraine': 203,
    'Gambian': 204, 'Bhutanese': 205, 'Guinean': 206
    }
LANGUAGE_map = {
    'English': 1, 'Spanish': 2, 'Dravidian': 3, 'German': 4, 'Hindi and related': 5, 'Japanese': 6, 'French': 7, 'Ukrainian, Ruthenian, Little Russian': 8, 'Other Persian dialects': 9,
    'Korean': 10, 'Greek': 11, 'Polish': 12, 'Arabic': 13, 'Chinese': 14, 'Vietnamese': 15, 'Italian': 16, 'Filipino, Tagalog': 17, 'Sub-Saharan Africa': 18, 'Tibetan': 19,
    'Portuguese': 20, 'Russian': 21, 'Dutch': 22, 'Indonesian': 23, 'Native': 24, 'Serbo-Croatian, Yugoslavian, Slavonian': 25, 'Rumanian': 26, 'Aleut, Eskimo': 27, 'Persian, Iranian, Farsi': 28,
    'Micronesian, Polynesian': 29, 'Thai, Siamese, Lao': 30, 'Navajo': 31, 'Aztecan, Nahuatl, Uto-Aztecan': 32, 'Near East Arabic dialect': 33, 'Athapascan': 34, 'Hebrew, Israeli': 35,
    'Other East/Southeast Asian': 36, 'Amharic, Ethiopian, etc.': 37, 'Burmese, Lisu, Lolo': 38, 'Czech': 39, 'Turkish': 40, 'Magyar, Hungarian': 41, 'Hamitic': 42, 'Other Balto-Slavic': 43,
    'Finnish': 44, 'Other Afro-Asiatic languages': 45, 'Hawaiian': 46, 'Iroquoian': 47, 'Armenian': 48, 'Other Altaic': 49, 'Other Malayan': 50, 'Celtic': 51, 'Lithuanian': 52,
    'Swedish': 53, 'Albanian': 54, 'Yiddish, Jewish': 55, 'Other or not reported': 56, 'Algonquian': 57, 'Norwegian': 58, 'Danish': 59, 'Slovak': 60, 'Muskogean': 61, 'Siouan languages': 62,
    'Keres': 63
    }
TRANWORK_map = {
    'Auto, truck, or van': 1, 'Walked only': 2, 'Worked at home': 3, 'Bus': 4, 'Bicycle': 5, 'Other': 6, 'Motorcycle': 7, 'Taxicab': 8, 'Ferryboat': 9,
    'Light rail, streetcar, or trolley (Carro público in PR)': 10, 'Subway or elevated': 11, 'Long-distance train or commuter train': 12
    }
degree_to_manual_label = {
    'Engineering': 0, 'Computer and Information Sciences': 1, 'Mathematics and Statistics': 2, 'Business': 3, 'Law': 4, 'Architecture': 5,
    'Physical Sciences': 6, 'Medical and Health Sciences and Services': 7, 'Biology and Life Sciences': 8, 'Environment and Natural Resources': 9,
    'Social Sciences': 10, 'Public Affairs, Policy, and Social Work': 11, 'Psychology': 12, 'Education Administration and Teaching': 13,
    'Communications': 14, 'Linguistics and Foreign Languages': 15, 'English Language, Literature, and Composition': 16, 'History': 17,
    'Area, Ethnic, and Civilization Studies': 18, 'Interdisciplinary and Multi-Disciplinary Studies (General)': 19, 'Fine Arts': 20,
    'Physical Fitness, Parks, Recreation, and Leisure': 21, 'Family and Consumer Sciences': 22, 'Agriculture': 23,
    'Philosophy and Religious Studies': 24, 'Theology and Religious Vocations': 25, 'Library Science': 26,
    'Criminal Justice and Fire Protection': 27, 'Engineering Technologies': 28, 'Construction Services': 29,
    'Transportation Sciences and Technologies': 30, 'Electrical and Mechanic Repairs and Technologies': 31,
    'Nuclear, Industrial Radiology, and Biological Technologies': 32, 'Communication Technologies': 33,
    'Cosmetology Services and Culinary Arts': 34, 'Military Technologies': 35, 'NAN': -1
}
speakeng_to_label = {
    'Yes, speaks only English': 0, 'Yes, speaks very well': 1, 'Yes, speaks well': 2,
    'Yes, but not well': 3, 'Does not speak English': 4
}
educ_to_label = {
    '5+ years of college': 0, '4 years of college': 1, '2 years of college': 2, '1 year of college': 3,
    'Grade 12': 4, 'Grade 11': 5, 'Grade 10': 6, 'Grade 9': 7,
    'Grade 5, 6, 7, or 8': 8, 'Nursery school to grade 4': 9, 'N/A or no schooling': 10
}
classwkr_map = {'Works for wages':0, 'Self-employed':1}



# Load components
with open("xgb_model_with_loo.pkl", "rb") as f:
    model_bundle = pickle.load(f)
average_mae = joblib.load("average_mae.pkl")
degree_encoder = joblib.load("degree_encoder.pkl")

# Extract the model and encoder
model = model_bundle["model"]
loo_encoder = model_bundle["loo_encoder"]
feature_cols = model_bundle["feature_cols"]


# Load Excel data for industry and occupation codes
@st.cache
def load_data():
    file_path = 'Mapping.xlsx'  # Adjust this path to your actual file
    industry_df = pd.read_excel(file_path, sheet_name='IND')
    occupation_df = pd.read_excel(file_path, sheet_name='OCC')
    return industry_df, occupation_df

industry_df, occupation_df = load_data()

st.title("Predicted Personal Income")

# Define input form
with st.form("income_form"):
    st.write("### Enter Person's Information")
    
    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input("Age", 0, 120, 30)
        sex = st.selectbox("Sex", ["Male", "Female"])
        state_name = st.selectbox("State", state_names)
        nchil = st.number_input("Number of Children", 0, 9, 0)
        uhrswork = st.number_input("Hours Worked per Week", 0, 100, 40)

    with col2:
        classwkrd = st.selectbox("Class of Worker", [
            "Local govt employee", "Self-employed, incorporated", "Self-employed, not incorporated",
            "State govt employee", "Wage/salary at non-profit", "Wage/salary, private"
        ])
        marst = st.selectbox("Marital Status", [
            "Married, spouse absent", "Married, spouse present", "Never married/single",
            "Separated", "Widowed"
])
        trantime = st.number_input("Transit Time (minutes)", 0, 999, 30)
        degree_choices = list(degree_to_manual_label.keys())
        selected_degrees = st.multiselect(
            "Select up to two Degree Fields",
            options=degree_choices,
            max_selections=2
        )

        # Pad with None if fewer than 2 selected
        deg1 = selected_degrees[0] if len(selected_degrees) > 0 else None
        deg2 = selected_degrees[1] if len(selected_degrees) > 1 else None

        speakeng = st.selectbox("English Proficiency (Encoded)", list(speakeng_to_label.keys()))
        speakeng_code = speakeng_to_label[speakeng]  # Apply English proficiency mapping here
        educ = st.selectbox("Education Level (Encoded)", list(educ_to_label.keys()))
        educ_code = educ_to_label[educ]

    with col3:
        race = st.selectbox("Race", list(RACE_map.keys()))
        race_code = RACE_map[race]
        bpl = st.selectbox("Birthplace Code (BPL)", list(BPL_map.keys())) 
        bpl_code = BPL_map[bpl]  # Apply birthplace mapping here
        ancestr1 = st.selectbox("Ancestry Code", list(ANCESTR1_map.keys()))
        ancestr1_code = ANCESTR1_map[ancestr1]  # Apply ancestry mapping here
        
        # Replace occsoc and ind with dropdowns based on the Excel file
        industry_options = industry_df['Industry Name'].tolist()
        occupation_options = occupation_df['Occupation Name'].tolist()

        selected_industry = st.selectbox("Select an Industry", industry_options)
        selected_occupation = st.selectbox("Select an Occupation", occupation_options)

        # Get the corresponding codes based on user selection
        ind = industry_df[industry_df['Industry Name'] == selected_industry]['Industry Code'].values[0]
        occsoc = occupation_df[occupation_df['Occupation Name'] == selected_occupation]['Occupation Code'].values[0]

        wkswork1 = st.number_input("Weeks Worked Last Year", 1, 52, 48)

    submitted = st.form_submit_button("Predict Income")

# Base input dictionary
input_dict = {
    "NCHILD": nchil,
    "AGE": age,
    "RACE": race_code,
    "BPL": bpl_code,
    "ANCESTR1": ancestr1_code,
    "WKSWORK1": wkswork1,
    "UHRSWORK": uhrswork,
    "TRANTIME": trantime,
    "SEX_Male": int(sex == "Female"),  # CORRECT for model trained with Male=0
    'CLASSWKRD_Local govt employee': int(classwkrd == "Local govt employee"),
    'CLASSWKRD_Self-employed, incorporated': int(classwkrd == "Self-employed, incorporated"),
    'CLASSWKRD_Self-employed, not incorporated': int(classwkrd == "Self-employed, not incorporated"),
    'CLASSWKRD_State govt employee': int(classwkrd == "State govt employee"),
    'CLASSWKRD_Wage/salary at non-profit': int(classwkrd == "Wage/salary at non-profit"),
    'CLASSWKRD_Wage/salary, private': int(classwkrd == "Wage/salary, private"),
    'MARST_Married, spouse absent': int(marst == "Married, spouse absent"),
    'MARST_Married, spouse present': int(marst == "Married, spouse present"),
    'MARST_Never married/single': int(marst == "Never married/single"),
    'MARST_Separated': int(marst == "Separated"),
    'MARST_Widowed': int(marst == "Widowed"),
    "SPEAKENG_ENCODED": speakeng_code,
    "EDUC_ENCODED": educ_code,
    "OCCSOC": occsoc,  # Occupation code
    "IND": ind,        # Industry code
    "STATEFIP": state_name,
}

# Create the input dataframe
input_df = pd.DataFrame([input_dict])

# The list of feature columns in the exact order expected by the model (this matches your provided order)
feature_order = [
    'NCHILD', 'AGE', 'RACE', 'BPL', 'ANCESTR1', 'WKSWORK1', 'UHRSWORK', 'TRANTIME', 'SEX_Male', 
    'CLASSWKRD_Local govt employee', 'CLASSWKRD_Self-employed, incorporated', 'CLASSWKRD_Self-employed, not incorporated', 
    'CLASSWKRD_State govt employee', 'CLASSWKRD_Wage/salary at non-profit', 'CLASSWKRD_Wage/salary, private', 
    'MARST_Married, spouse absent', 'MARST_Married, spouse present', 'MARST_Never married/single', 'MARST_Separated', 
    'MARST_Widowed', 'nan', 'Theology and Religious Vocations', 'Linguistics and Foreign Languages', 
    'Public Affairs, Policy, and Social Work', 'Engineering', 'Computer and Information Sciences', 'Mathematics and Statistics', 
    'Environment and Natural Resources', 'Business', 'Psychology', 'Biology and Life Sciences', 
    'Education Administration and Teaching', 'Physical Sciences', 'Medical and Health Sciences and Services', 
    'Social Sciences', 'Engineering Technologies', 'Criminal Justice and Fire Protection', 'Family and Consumer Sciences', 
    'Liberal Arts and Humanities', 'English Language, Literature, and Composition', 'Area, Ethnic, and Civilization Studies', 
    'Interdisciplinary and Multi-Disciplinary Studies (General)', 'Physical Fitness, Parks, Recreation, and Leisure', 
    'Communications', 'Fine Arts', 'History', 'Construction Services', 'Transportation Sciences and Technologies', 'Agriculture', 
    'Philosophy and Religious Studies', 'Architecture', 'Communication Technologies', 'Cosmetology Services and Culinary Arts', 
    'Law', 'Electrical and Mechanic Repairs and Technologies', 'Nuclear, Industrial Radiology, and Biological Technologies', 
    'Library Science', 'Military Technologies', 'SPEAKENG_ENCODED', 'EDUC_ENCODED', 'OCCSOC', 'IND', 'STATEFIP'
]

# 1. Create degree input DataFrame
degree_input_df = pd.DataFrame([{
    'DEGFIELD': deg1,  # Replace deg1 with actual value
    'DEGFIELD2': deg2  # Replace deg2 with actual value
}])

# 2. Get multi-hot encoded degrees and wrap in DataFrame
degree_array = degree_encoder.transform(degree_input_df.values)
degree_features_df = pd.DataFrame(degree_array, columns=degree_encoder.classes_)

# 3. Ensure degree columns are in the correct order in feature_order
degree_columns = degree_features_df.columns.tolist()

# 4. Add degree columns to the input DataFrame
input_df = input_df.join(degree_features_df)

# 5. Reorder the input DataFrame to match the feature order
missing_columns = set(feature_order) - set(input_df.columns)

# Add missing columns with default values (0 or NaN)
for col in missing_columns:
    input_df[col] = 0  # or pd.NA if you'd prefer

# Reorder the columns of input_df to match feature_order
input_df = input_df[feature_order]

# Apply Leave-One-Out Encoding (LOO) on the relevant columns if needed
user_input_loo = input_df[feature_cols]  # Use the feature columns for LOO encoding
user_input_non_loo = input_df.drop(columns=feature_cols)

# Transform LOO columns using the fitted encoder
user_input_loo_encoded = loo_encoder.transform(user_input_loo)

# Recombine the transformed LOO columns with the non-LOO columns
user_input_transformed = pd.concat([user_input_non_loo.reset_index(drop=True),
                                    user_input_loo_encoded.reset_index(drop=True)], axis=1)

# Ensure the columns match exactly with what the model expects
user_input_transformed = user_input_transformed[model.feature_names_in_]


# Predict income using the trained model
predicted_income = model.predict(user_input_transformed)[0]

# Calculate the income range based on the average MAE
lower = predicted_income - average_mae
upper = predicted_income + average_mae

# Counterfactual analysis (opposite gender toggle)
opposite_input = user_input_transformed.copy()
if 'SEX_Male' in opposite_input.columns:
    opposite_input['SEX_Male'] = 1 - opposite_input['SEX_Male']

# Predict counterfactual income
opposite_income = model.predict(opposite_input)[0]
opp_lower = opposite_income - average_mae
opp_upper = opposite_income + average_mae

# Calculate percent difference
percent_diff = ((predicted_income - opposite_income) / opposite_income) * 100

# DEBUG INFO
#st.write("DEBUG - Gender encoding:")
#st.write("Original (0=Male, 1=Female):")
#st.write(user_input_transformed[['SEX_Male']])
#st.write("Counterfactual:")
#st.write(opposite_input[['SEX_Male']])

# Determine gender from encoding (0=Male, 1=Female)
if user_input_transformed['SEX_Male'].iloc[0] == 1:
    gender_label = "Female"
    opp_gender_label = "Male"
else:
    gender_label = "Male"
    opp_gender_label = "Female"

# Display results
st.subheader("Estimated Annual Income")
st.success(f"{gender_label}: ${predicted_income:,.0f} (±${average_mae:,.0f})")
st.write(f"**Range:** ${lower:,.0f} - ${upper:,.0f}")

st.subheader("Counterfactual (Opposite Gender)")
st.info(f"{opp_gender_label}: ${opposite_income:,.0f} (±${average_mae:,.0f})")
st.write(f"**Range:** ${opp_lower:,.0f} - ${opp_upper:,.0f}")

st.markdown(
    f"**The predicted income is {abs(percent_diff):.1f}% "
    f"{'higher' if percent_diff > 0 else 'lower'} than it would be if the person were {opp_gender_label}.**"
)
