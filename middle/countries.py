"""Middle task 4: countries with capital, continent and official language (middle/PLAN.md).

A country is kept only if all three facts are unambiguous. Rules, applied in order; every
excluded country is listed in EXCLUDED with the rule that removed it:

  R1 language  more than one official language at national level (regional or "national
               language" status below the official one does not count), or no official language
               and no overwhelmingly dominant one, or a recent change of official status.
  R2 capital   more than one capital (constitutional, administrative, judicial), a capital that
               is disputed, or one being moved by law.
  R3 status    disputed or limited recognition as a state.
  R4 continent territory in two continents in common usage, or in Oceania (whose continent is
               named "Australia" or "Oceania" interchangeably). Continents used: Africa, Asia,
               Europe, North America (incl. Central America and the Caribbean), South America.
  R5 name      the country name usually takes "the" (so the answer would start with " the"), or
               has two common English names.
  R6 leak      the capital's name contains the country's name (the first hop is given away), or
               the country is a city-state.
  R7 answer    the language has two common English names (e.g. Persian / Farsi).

The table is written from general knowledge (status as of 2025). It is about 90 rows rather
than the plan's "about 120", because the rules above remove many countries; the rules were not
relaxed to reach the target.
"""

# (country, capital, continent, language)
ROWS = [
    # Europe
    ("France", "Paris", "Europe", "French"),
    ("Germany", "Berlin", "Europe", "German"),
    ("Italy", "Rome", "Europe", "Italian"),
    ("Spain", "Madrid", "Europe", "Spanish"),
    ("Portugal", "Lisbon", "Europe", "Portuguese"),
    ("Poland", "Warsaw", "Europe", "Polish"),
    ("Hungary", "Budapest", "Europe", "Hungarian"),
    ("Austria", "Vienna", "Europe", "German"),
    ("Greece", "Athens", "Europe", "Greek"),
    ("Sweden", "Stockholm", "Europe", "Swedish"),
    ("Norway", "Oslo", "Europe", "Norwegian"),
    ("Denmark", "Copenhagen", "Europe", "Danish"),
    ("Iceland", "Reykjavik", "Europe", "Icelandic"),
    ("Slovakia", "Bratislava", "Europe", "Slovak"),
    ("Croatia", "Zagreb", "Europe", "Croatian"),
    ("Serbia", "Belgrade", "Europe", "Serbian"),
    ("Albania", "Tirana", "Europe", "Albanian"),
    ("Bulgaria", "Sofia", "Europe", "Bulgarian"),
    ("Romania", "Bucharest", "Europe", "Romanian"),
    ("Ukraine", "Kyiv", "Europe", "Ukrainian"),
    ("Lithuania", "Vilnius", "Europe", "Lithuanian"),
    ("Latvia", "Riga", "Europe", "Latvian"),
    ("Estonia", "Tallinn", "Europe", "Estonian"),
    ("Liechtenstein", "Vaduz", "Europe", "German"),
    ("Andorra", "Andorra la Vella", "Europe", "Catalan"),
    # Asia
    ("Japan", "Tokyo", "Asia", "Japanese"),
    ("South Korea", "Seoul", "Asia", "Korean"),
    ("North Korea", "Pyongyang", "Asia", "Korean"),
    ("Mongolia", "Ulaanbaatar", "Asia", "Mongolian"),
    ("Vietnam", "Hanoi", "Asia", "Vietnamese"),
    ("Thailand", "Bangkok", "Asia", "Thai"),
    ("Cambodia", "Phnom Penh", "Asia", "Khmer"),
    ("Bhutan", "Thimphu", "Asia", "Dzongkha"),
    ("Nepal", "Kathmandu", "Asia", "Nepali"),
    ("Saudi Arabia", "Riyadh", "Asia", "Arabic"),
    ("Oman", "Muscat", "Asia", "Arabic"),
    ("Qatar", "Doha", "Asia", "Arabic"),
    ("Bahrain", "Manama", "Asia", "Arabic"),
    ("Jordan", "Amman", "Asia", "Arabic"),
    ("Syria", "Damascus", "Asia", "Arabic"),
    ("Lebanon", "Beirut", "Asia", "Arabic"),
    ("Uzbekistan", "Tashkent", "Asia", "Uzbek"),
    ("Turkmenistan", "Ashgabat", "Asia", "Turkmen"),
    ("Tajikistan", "Dushanbe", "Asia", "Tajik"),
    ("Brunei", "Bandar Seri Begawan", "Asia", "Malay"),
    # Africa
    ("Egypt", "Cairo", "Africa", "Arabic"),
    ("Libya", "Tripoli", "Africa", "Arabic"),
    ("Tunisia", "Tunis", "Africa", "Arabic"),
    ("Mauritania", "Nouakchott", "Africa", "Arabic"),
    ("Senegal", "Dakar", "Africa", "French"),
    ("Guinea", "Conakry", "Africa", "French"),
    ("Ghana", "Accra", "Africa", "English"),
    ("Togo", "Lomé", "Africa", "French"),
    ("Nigeria", "Abuja", "Africa", "English"),
    ("Gabon", "Libreville", "Africa", "French"),
    ("Angola", "Luanda", "Africa", "Portuguese"),
    ("Mozambique", "Maputo", "Africa", "Portuguese"),
    ("Zambia", "Lusaka", "Africa", "English"),
    ("Malawi", "Lilongwe", "Africa", "English"),
    ("Botswana", "Gaborone", "Africa", "English"),
    ("Namibia", "Windhoek", "Africa", "English"),
    ("Liberia", "Monrovia", "Africa", "English"),
    ("Sierra Leone", "Freetown", "Africa", "English"),
    ("South Sudan", "Juba", "Africa", "English"),
    # North America
    ("Belize", "Belmopan", "North America", "English"),
    ("Honduras", "Tegucigalpa", "North America", "Spanish"),
    ("El Salvador", "San Salvador", "North America", "Spanish"),
    ("Nicaragua", "Managua", "North America", "Spanish"),
    ("Costa Rica", "San José", "North America", "Spanish"),
    ("Cuba", "Havana", "North America", "Spanish"),
    ("Jamaica", "Kingston", "North America", "English"),
    ("Trinidad and Tobago", "Port of Spain", "North America", "English"),
    ("Barbados", "Bridgetown", "North America", "English"),
    ("Grenada", "St. George's", "North America", "English"),
    ("Saint Lucia", "Castries", "North America", "English"),
    ("Antigua and Barbuda", "St. John's", "North America", "English"),
    ("Saint Kitts and Nevis", "Basseterre", "North America", "English"),
    ("Saint Vincent and the Grenadines", "Kingstown", "North America", "English"),
    # South America
    ("Peru", "Lima", "South America", "Spanish"),
    ("Argentina", "Buenos Aires", "South America", "Spanish"),
    ("Chile", "Santiago", "South America", "Spanish"),
    ("Colombia", "Bogotá", "South America", "Spanish"),
    ("Venezuela", "Caracas", "South America", "Spanish"),
    ("Ecuador", "Quito", "South America", "Spanish"),
    ("Uruguay", "Montevideo", "South America", "Spanish"),
    ("Brazil", "Brasília", "South America", "Portuguese"),
    ("Guyana", "Georgetown", "South America", "English"),
    ("Suriname", "Paramaribo", "South America", "Dutch"),
]

EXCLUDED = {
    # R1 language
    "Finland": "R1 (Finnish and Swedish)", "Ireland": "R1 (Irish and English)",
    "Belgium": "R1", "Switzerland": "R1", "Luxembourg": "R1", "Malta": "R1 (Maltese, English)",
    "Belarus": "R1 (Belarusian, Russian)", "Bosnia and Herzegovina": "R1",
    "North Macedonia": "R1 (Macedonian, Albanian)", "Moldova": "R1 (Moldovan renamed Romanian, 2023)",
    "Canada": "R1", "Haiti": "R1 (French, Haitian Creole)", "Paraguay": "R1 (Spanish, Guarani)",
    "Bolivia": "R1 and R2 (Sucre / La Paz)", "India": "R1", "Pakistan": "R1 (Urdu, English)",
    "Afghanistan": "R1 (Pashto, Dari)", "Iraq": "R1 (Arabic, Kurdish)",
    "Kyrgyzstan": "R1 (Kyrgyz, Russian)", "Algeria": "R1 (Arabic, Tamazight)",
    "Morocco": "R1 (Arabic, Tamazight)", "Sudan": "R1 (Arabic, English)",
    "Cameroon": "R1", "Chad": "R1", "Kenya": "R1 (Swahili, English)",
    "Uganda": "R1 (English, Swahili)", "Rwanda": "R1", "Burundi": "R1", "Zimbabwe": "R1",
    "Somalia": "R1 (Somali, Arabic)", "Djibouti": "R1", "Eritrea": "R1 (no official language)",
    "Ethiopia": "R1 (several federal working languages since 2020)", "Madagascar": "R1",
    "Lesotho": "R1", "Eswatini": "R1", "Seychelles": "R1", "Comoros": "R1",
    "Mauritius": "R1 (no de jure official language)", "Equatorial Guinea": "R1",
    "Mali": "R1 (French demoted to working language, 2023)",
    "Burkina Faso": "R1 (French status changed, 2023)", "Niger": "R1 (status changed, 2025)",
    "United States": "R1 (official language only by 2025 executive order) and R5",
    "Singapore": "R1 and R6", "Philippines": "R1 and R5", "Timor-Leste": "R1 and R5",
    # R2 capital
    "Netherlands": "R2 (Amsterdam / The Hague) and R5", "South Africa": "R2 (three capitals)",
    "Sri Lanka": "R2", "Malaysia": "R2 (Kuala Lumpur / Putrajaya)",
    "Tanzania": "R2 (Dodoma / Dar es Salaam)", "Myanmar": "R2 (Naypyidaw / Yangon)",
    "Benin": "R2 (Porto-Novo / Cotonou)", "Ivory Coast": "R2 (Yamoussoukro / Abidjan) and R5",
    "Indonesia": "R2 (move to Nusantara by law)", "Yemen": "R2 (Sanaa / Aden)",
    "Israel": "R2", "Montenegro": "R2 (Podgorica / Cetinje as old royal capital)",
    # R3 status
    "Taiwan": "R3", "Kosovo": "R3", "Palestine": "R3", "Western Sahara": "R3",
    # R4 continent
    "Russia": "R4", "Turkey": "R4", "Kazakhstan": "R4 and R1", "Georgia": "R4",
    "Armenia": "R4", "Azerbaijan": "R4", "Cyprus": "R4", "Australia": "R4",
    "New Zealand": "R4", "Fiji": "R4", "Papua New Guinea": "R4", "Samoa": "R4", "Tonga": "R4",
    # R5 name
    "United Kingdom": "R5", "United Arab Emirates": "R5", "Czech Republic": "R5 (Czechia)",
    "Dominican Republic": "R5", "Bahamas": "R5", "Gambia": "R5",
    "Central African Republic": "R5 and R1", "Democratic Republic of the Congo": "R5 and R1",
    "Republic of the Congo": "R5", "Cape Verde": "R5 (Cabo Verde)",
    "Dominica": "R5 (confused with the Dominican Republic)",
    # R6 leak
    "Mexico": "R6 (Mexico City)", "Guatemala": "R6 (Guatemala City)",
    "Panama": "R6 (Panama City)", "Kuwait": "R6 (Kuwait City)", "Guinea-Bissau": "R6 (Bissau)",
    "Sao Tome and Principe": "R6", "San Marino": "R6", "Monaco": "R6", "Vatican City": "R6",
    # R7 answer
    "China": "R7 (Chinese / Mandarin)", "Iran": "R7 (Persian / Farsi)",
    "Bangladesh": "R7 (Bengali / Bangla)", "Laos": "R7 (Lao / Laotian)",
    "Maldives": "R7 (Dhivehi / Maldivian)", "Slovenia": "R7 (Slovene / Slovenian)",
}

# Rows used only as the three worked examples in front of every template (never as items).
EXAMPLES = ["France", "Japan", "Argentina"]
