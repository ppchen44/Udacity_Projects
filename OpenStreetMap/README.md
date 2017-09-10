# OpenStreetMap Data Case Study
## Map Area
San Jose, CA, United States

- https://mapzen.com/data/metro-extracts/
- https://mapzen.com/data/metro-extracts/metro/san-jose_california/

**I am living San Jose right now, so I want to know what kind of data structure it is and more detail informations of the city.**

## Problem Encountered in the xml map data
After I had downloaded the data and generated a small size of data as sample, I run my Parsing_xml_Data.py. I found there are several problems.
- Abbreviated street names.Mt 'Hamilton Rd'
- Postcode with hype.
- Wrong postcodes which do not belong to San Jose
- Overrite the same informations in tags
- Data which extracted from Tiger GPS 

## Abbreviated street names
After I had run my Parsing_xml_Data.py on my sample data, I got 'uncorrect' form of address such as ```Fountain Oaks Dr```, ```Mt Hamilton Rd``` . To deal with those type of addresses, I used function which was mentioned in Case study: OpenStreetMap Data.
```
def update_name(name, mapping):
    m = street_type_re.search(name)
    street_type = m.group()
    if street_type not in expected:
        if street_type in mapping.keys():
            name = re.sub(street_type, mapping[m.group()], name)
    return name
 ```
 ```Mt Hamilton Rd => Mt Hamilton Road```

However, There was another type of address: ```'Zanker Rd., San Jose, CA'```. In order to correct this, I wrote another the below function. After I had run **audit_CA_addr()**, I got ```print 'Zanker Rd., San Jose, CA:','=>', audit_CA_addr('Zanker Rd., San Jose, CA',mapping)```
 ```
 def audit_CA_addr(name, mapping):
    m = street_type_re.search(name)
    street_type = m.group()
    if street_type == 'CA':
        addr_info = name.split(',')[0]
        name = update_name(addr_info, mapping)
    return name
 ```
``` Zanker Rd., San Jose, CA: => Zanker Road```
 ## Postcode with hype and correct 'wrong' postcode
One of address types is postcode, however, some values of postcode with hype in it. I added a if statement to find out whether the postcode value includes hype ```if re.search(hyphen, child.attrib['v']):```. Then I split this value of postcode by using ```child.attrib['v'].split('-')[0]```
```
                        elif child.attrib['k'].split(':', 1)[1] == 'postcode':
                            if re.search(hyphen, child.attrib['v']):
                                if child.attrib['v'].split('-')[0] not in zipcode_san_jose:
                                    continue
                                else:
                                    node_tags_dic['id'] = element.attrib['id']
                                    node_tags_dic['key'] = child.attrib['k'].split(':', 1)[1]
                                    node_tags_dic['type'] = child.attrib['k'].split(':', 1)[0]
                                    node_tags_dic['value'] = child.attrib['v'].split('-')[0]
                            else:
                                if child.attrib['v'] not in zipcode_san_jose:
                                    continue
                                else:
                                    node_tags_dic['id'] = element.attrib['id']
                                    node_tags_dic['key'] = child.attrib['k'].split(':', 1)[1]
                                    node_tags_dic['type'] = child.attrib['k'].split(':', 1)[0]
                                    node_tags_dic['value'] = child.attrib['v']
```
## Overrite the same informations in tags
Street names in the second level 'k' tags pulled from Tiger GPS data and divided into segments.
```if child.attrib['k'].split(':', 1)[0] == 'tiger':```

## Data Overview
### Files size
san-jose_california.osm.............368.3MB                   
nodes.csv.............................141.1MB             
nodes_tags.csv.........................3MB                     
ways.csv............................13.8MB                      
ways_tags.csv.......................19.6MB                     
ways_nodes.csv......................47.1MB                       

### Number of nodes
```sqlite> SELECT COUNT(*) FROM nodes;```

1692033

### Number of ways
```sqlite> SELECT COUNT(*) FROM ways;```

232001

### Number of unique users
```sqlite> SELECT COUNT(DISTINCT (e.uid)) FROM (SELECT uid FROM nodes UNION ALL SELECT uid FROM ways) e;```

1374

### TOP 10 contributing users
```
sqlite> SELECT e.user, COUNT(*) as num
   FROM (SELECT user FROM nodes UNION ALL SELECT user FROM ways) e
   GROUP BY e.user
   ORDER BY num DESC
   LIMIT 10;
```
```
andygol|295555
nmixter|284851
mk408|147154
Bike Mapper|91059
samely|81073
RichRico|76177
dannykath|74426
MustangBuyer|65038
karitotp|63431
Minh Nguyen|52974
```
### Top informations of San Jose
```
sqlite> SELECT e.key, COUNT(*) as num FROM
   (SELECT key FROM nodes_tags UNION ALL SELECT key FROM ways_tags) e
   GROUP BY e.key
   ORDER BY num DESC
   LIMIT 20;
```
```
building|137091
highway|90294
name|49367
county|32494
cfcc|25619
housenumber|22213
street|21819
oneway|16524
service|16412
lanes|15655
reviewed|14007
source|12213
height|10973
maxspeed|10903
amenity|8052
surface|7636
cycleway|6262
city|5978
waterway|5653
layer|5461
```

### Top 10 Amenities
```
sqlite> SELECT e.value, COUNT(*) as num
   FROM (SELECT value FROM nodes_tags WHERE key = 'amenity' UNION ALL SELECT value FROM ways_tags WHERE key = 'amenity') e
   GROUP BY e.value
   ORDER BY num DESC
   LIMIT 10;
```
```
parking|2174
restaurant|1050
fast_food|534
school|533
place_of_worship|354
bench|332
cafe|271
fuel|247
bicycle_parking|212
toilets|205
```

## Additional Ideas
In this map dataset, some informations are coming from tiger GPS which do not have the same format with others. Some postcodes do not belong to San Jose, but they appear in the dataset, which indicate OpenStreetMap data have some potential problems with district boundary. So many values with different format from others which make the data wrangling process more complicate. In the real word, we might need to take more time to clean data so that they can be used in further processes.

### Anticipated problems and benefits
- Improve 1: keys like 'amenity', 'suisine', 'name' could be formated in dictionary
```
    { 
        'amenity': {'restaurant':[{'cuisine': ...,
                                    'name': ...}, {'cuisine': ..., 'name': ...}, ...],
                    'bus_station': [{'network': ...}],
                    ....}
     }
```
     . Pros: It makes data logically structured and meanwhile keep all information of amenity
     
     . Cons: It might take more time to parse the data
     
- Improve 2: keys like 'addr:country', 'addr:state' should be transformed into 
```
    {
        'addr': {'city': ...,
                 'country': ...,
                 'state': ...}
        ...
    }
```
    . Pros: It makes data more readable and more logical structured and keep all information of address.
    
    . Cons: It might take more time to parse the data. 

- Helping with marketing decisions

We could investigate foods' type, name and address or coffee houses' type, name and address if we want to open another restaurant or coffee house. After investigating all these informations, we could know residents who live in San Jose like which kind of food or drinks. For example, after I had investigated coffee houses' name and how many each coffee house are in San Jose area, I would know Starbucks Coffee has higher market share than Peet's Coffee. According to the result of the most popular cuisines, we could know which type of food is welcomed by residents. All these informations would help us to make a marketing decision about restaurant. 
#### Names of coffee house in San Jose
```
sqlite> SELECT nodes_tags.value, COUNT(*) as num
   FROM nodes_tags 
   JOIN (SELECT DISTINCT(id) FROM nodes_tags WHERE value LIKE '%coffe%') i
   ON nodes_tags.id=i.id
   WHERE nodes_tags.key='name'
   GROUP BY nodes_tags.value
   ORDER BY num DESC
   LIMIT 5;
```
```
Starbucks|49
Peet's Coffee & Tea|6
Starbucks Coffee|5
Peet's Coffee|3
Philz Coffee|3
```
#### Most popular cuisines
```
sqlite> SELECT nodes_tags.value, COUNT(*) as num
   FROM nodes_tags 
   JOIN (SELECT DISTINCT(id) FROM nodes_tags WHERE value='restaurant') i
   ON nodes_tags.id=i.id
   WHERE nodes_tags.key='cuisine'
   GROUP BY nodes_tags.value
   ORDER BY num DESC
   LIMIT 10;
```
```
vietnamese|74
chinese|65
mexican|61
pizza|56
japanese|43
italian|31
indian|30
american|28
thai|27
sushi|23
```
- Simplify and correct the informations which are extracted from GPS source
Many informations are coming from GPS which are not prepared to further analysis. Write functions to clear it and correct it.

## Conclusion
After I had reviewed the data, I found that there are still have more data wrangling works to do. I have learned data extracting, wrangling, importing csv to sql database and manipulating with SQL. This is the best practice for my further works about data mining. 









