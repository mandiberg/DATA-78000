'''
These are some sample queries that get country level data from Wikidata
Use these as examples to think about, and models to work from, as you figure out where to explore.

note: I gave the file a SQL suffix, even thought it is not actually SQL.
Giving it that suffix will color code the query in your IDE for easier reading

'''


-- NUMERICAL DATA

SELECT ?country ?countryLabel 
       (MAX(?population) AS ?pop) 
       (SAMPLE(?area) AS ?areaKm2)
       (MAX(?gdp) AS ?gdpUSD)
       (MAX(?lifeExpectancy) AS ?lifeExp)
WHERE {
  ?country wdt:P31 wd:Q3624078.
  
  OPTIONAL { ?country wdt:P1082 ?population. }
  OPTIONAL { ?country wdt:P2046 ?area. }
  OPTIONAL { ?country wdt:P2131 ?gdp. }
  OPTIONAL { ?country wdt:P2250 ?lifeExpectancy. }
  
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
GROUP BY ?country ?countryLabel
ORDER BY ?countryLabel



-- CATEGORICAL DATA

SELECT ?country ?countryLabel 
       (GROUP_CONCAT(DISTINCT ?governmentFormLabel; separator="|") AS ?governmentForms)
       (GROUP_CONCAT(DISTINCT ?continentLabel; separator="|") AS ?continents)
       (SAMPLE(?drivingSideLabel) AS ?drivingSide)
       (GROUP_CONCAT(DISTINCT ?religionLabel; separator="|") AS ?religions)
       (IF(BOUND(?landlockedCheck), "landlocked", "coastal") AS ?landlockedStatus)
WHERE {
  ?country wdt:P31 wd:Q3624078.   # sovereign state
  
  OPTIONAL { 
    ?country wdt:P122 ?governmentForm. 
    ?governmentForm rdfs:label ?governmentFormLabel. 
    FILTER(LANG(?governmentFormLabel) = "en")
  }
  OPTIONAL { 
    ?country wdt:P30 ?continent. 
    ?continent rdfs:label ?continentLabel. 
    FILTER(LANG(?continentLabel) = "en")
  }
  OPTIONAL { 
    ?country wdt:P1622 ?drivingSide. 
    ?drivingSide rdfs:label ?drivingSideLabel. 
    FILTER(LANG(?drivingSideLabel) = "en")
  }
  OPTIONAL { 
    ?country wdt:P140 ?religion. 
    ?religion rdfs:label ?religionLabel. 
    FILTER(LANG(?religionLabel) = "en")
  }
  OPTIONAL { 
    ?country wdt:P31 ?landlockedCheck. 
    FILTER(?landlockedCheck = wd:Q179049) 
  }
  
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
GROUP BY ?country ?countryLabel ?landlockedCheck
ORDER BY ?countryLabel


-- LANGUAGES
-- You will have to do some post processing to make it work, but it might be useful to think about

SELECT ?country ?countryLabel ?officialLanguageLabel
       (GROUP_CONCAT(DISTINCT ?ancestorLabel; separator="|") AS ?languageLineage)
WHERE {
  ?country wdt:P31 wd:Q3624078.          # sovereign state
  ?country wdt:P37 ?officialLanguage.    # official language

  ?officialLanguage rdfs:label ?officialLanguageLabel .
  FILTER(LANG(?officialLanguageLabel) = "en")

  OPTIONAL {
    ?officialLanguage wdt:P279* ?ancestor .   # climb the "subclass of" chain
    ?ancestor rdfs:label ?ancestorLabel .
    FILTER(LANG(?ancestorLabel) = "en")
    FILTER(REGEX(?ancestorLabel, "languages$", "i"))  # keep only family-like ancestors
  }
}
GROUP BY ?country ?countryLabel ?officialLanguageLabel
ORDER BY ?countryLabel