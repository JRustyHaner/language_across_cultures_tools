#Description: This script downloads the PDF files from the UN website
# and saves them to a directory named after the country code.

import requests
import os

# List of UN country codes
country_codes = [
    "us", "br", "co", "jo", "pl", "cu", "tr", "pt", "qa", "za", "tm", "ua", "gt", "hu", "ch", "si", "uz",
    "bo", "kz", "ir", "dz", "ar", "sv", "kg", "py", "pe", "mz", "pa", "ng", "uy", "cz", "pw", "sn", "de",
    "jp", "sc", "rw", "cy", "na", "ro", "sr", "ba", "gh", "lt", "sk", "fi", "bg", "gy", "hr", "ec", "ao", "lv",
    "kr", "tj", "hn", "ee", "km", "do", "md", "sl", "mc", "cl", "mn", "mr", "lr", "cd", "sz", "mh", "bw", "it", "es",
    "st", "be", "lb", "ly", "vu", "sa", "gw", "al", "sd", "cn", "ci", "ug", "gm", "gq", "tz", "tt", "gr", "np",
    "kw", "fr", "ne", "ls", "kh", "ge", "ie", "gd", "tv", "ht", "to", "gb", "sg", "au", "bh", "bs", "se", "cr"
]
country_names = {
    "us": "United States",
    "br": "Brazil",
    "co": "Colombia",
    "jo": "Jordan",
    "pl": "Poland",
    "cu": "Cuba",
    "tr": "Turkey",
    "pt": "Portugal",
    "qa": "Qatar",
    "za": "South Africa",
    "tm": "Turkmenistan",
    "ua": "Ukraine",
    "gt": "Guatemala",
    "hu": "Hungary",
    "ch": "Switzerland",
    "si": "Slovenia",
    "uz": "Uzbekistan",
    "bo": "Bolivia (Plurinational State of)",
    "kz": "Kazakhstan",
    "ir": "Iran",
    "dz": "Algeria",
    "ar": "Argentina",
    "sv": "El Salvador",
    "kg": "Kyrgyzstan",
    "py": "Paraguay",
    "pe": "Peru",
    "mz": "Mozambique",
    "pa": "Panama",
    "ng": "Nigeria",
    "uy": "Uruguay",
    "cz": "Czechia",
    "pw": "Palau",
    "sn": "Senegal",
    "de": "Germany",
    "jp": "Japan",
    "sc": "Seychelles",
    "rw": "Rwanda",
    "cy": "Cyprus",
    "na": "Namibia",
    "ro": "Romania",
    "sr": "Suriname",
    "ba": "Bosnia and Herzegovina",
    "gh": "Ghana",
    "lt": "Lithuania",
    "sk": "Slovakia",
    "fi": "Finland",
    "bg": "Bulgaria",
    "gy": "Guyana",
    "hr": "Croatia",
    "ec": "Ecuador",
    "ao": "Angola",
    "lv": "Latvia",
    "kr": "South Korea",
    "tj": "Tajikistan",
    "hn": "Honduras",
    "ee": "Estonia",
    "km": "Comoros",
    "do": "Dominican Republic",
    "md": "Moldova",
    "sl": "Sierra Leone",
    "mc": "Monaco",
    "cl": "Chile",
    "mn": "Mongolia",
    "mr": "Mauritania",
    "lr": "Liberia",
    "cd": "Democratic Republic of the Congo",
    "sz": "Eswatini",
    "mh": "Marshall Islands",
    "bw": "Botswana",
    "it": "Italy",
    "es": "Spain",
    "st": "São Tomé and Príncipe",
    "be": "Belgium",
    "lb": "Lebanon",
    "ly": "Libya",
    "vu": "Vanuatu",
    "sa": "Saudi Arabia",
    "gw": "Guinea-Bissau",
    "al": "Albania",
    "sd": "Sudan",
    "cn": "China",
    "ci": "Côte d'Ivoire",
    "ug": "Uganda",
    "gm": "Gambia (Republic of The)",
    "gq": "Equatorial Guinea",
    "tz": "United Republic of Tanzania",
    "tt": "Trinidad and Tobago",
    "gr": "Greece",
    "np": "Nepal",
    "kw": "Kuwait",
    "fr": "France",
    "ne": "Niger",
    "ls": "Lesotho",
    "kh": "Cambodia",
    "ge": "Georgia",
    "ie": "Ireland",
    "gd": "Grenada",
    "tv": "Tuvalu",
    "ht": "Haiti",
    "to": "Tonga",
    "gb": "United Kingdom of Great Britain and Northern Ireland",
    "sg": "Singapore",
    "au": "Australia",
    "bh": "Bahrain",
    "bs": "Bahamas",
    "se": "Sweden",
    "cr": "Costa Rica",
    "vc": "Saint Vincent and the Grenadines",
    "ws": "Samoa",
    "kn": "Saint Kitts and Nevis",
    "cv": "Cabo Verde",
    "so": "Somalia",
    "la": "Lao People's Democratic Republic",
    "et": "Ethiopia",
    "pg": "Papua New Guinea",
    "az": "Azerbaijan",
    "ru": "Russian Federation",
    "id": "Indonesia",
    "mx": "Mexico",
    "ph": "Philippines",
    "nz": "New Zealand",
    "tn": "Tunisia",
    "is": "Iceland",
    "eg": "Egypt",
    "sa": "Saudi Arabia",
    "dj": "Djibouti",
    "by": "Belarus",
    "om": "Oman",
    "bz": "Belize",
    "er": "Eritrea",
    "am": "Armenia",
    "bf": "Burkina Faso",
    "bn": "Brunei Darussalam",
    "ve": "Venezuela (Bolivarian Republic of)",
    "no": "Norway",
    "ml": "Mali",
    "ae": "United Arab Emirates",
    "in": "India",
    "jm": "Jamaica",
    "bt": "Bhutan",
    "cm": "Cameroon",
    "zm": "Zambia",
    "va": "Holy See",
    "sy": "Syrian Arab Republic",
    "mv": "Maldives",
    "ni": "Nicaragua",
    "kp": "North Korea",
    "bj": "Benin",
    "sm": "San Marino",
    "ca": "Canada",
    "vu": "Vanuatu",
    "ma": "Morocco"
    # Add more country codes and names as needed
}


# Add more country codes as needed

# Directory path to save the PDF files
save_directory = "/media/rusty/Data2/UNGA/UNGA_78/pdf"

# URL template with a placeholder for the country code
url_template = "https://gadebate.un.org/sites/default/files/gastatements/[variable%3Acurrent_session]/{country_code}_en.pdf"

for country_code in country_codes:
    url = url_template.format(country_code=country_code)

    try:
        response = requests.get(url)
        if response.status_code == 200:
            #get the country's name
            country_name = country_names[country_code]
            # Content has been successfully downloaded
            pdf_file_path = os.path.join(save_directory, f"{country_name}.pdf")
            with open(pdf_file_path, "wb") as pdf_file:
                pdf_file.write(response.content)
            print(f"Downloaded {country_code} to {pdf_file_path}")
        else:
            print(f"Failed to download the content for {country_code}. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while downloading {country_code}: {e}")
