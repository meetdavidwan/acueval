from model import StableBeluga
import sys
from tqdm import tqdm
import json

lm = StableBeluga()

acu_generation_prompt = """Please breakdown the following passage into independent facts: Theme of film is children and features parents talking about their offspring . PM says what he wants for his own children, he wants for every child in UK . Broadcast is first of five to be released over course of election campaign .
- Theme of film is children and features parents talking about their offspring.
- PM says what he wants for his own children, he wants for every child in UK.
- Broadcast is first of five.
- Broadcasts will be released over course of election campaign.

Please breakdown the following passage into independent facts: Chelsea boss Jose Mourinho says Paris Saint-Germain are the most aggressive side his team have played this season . Blues host French giants in Champions League last-16 second leg . Laurent Blanc also claims Chelsea have 'dirty tricks' with Diego Costa . Chelsea have committed more fouls than PSG in the competition so far . David Luiz proved he had a ruthless streak in him in last leg in Paris . Thiago Silva, Marco Verratti and Zlatan Ibrahimovic are other danger men . CLICK HERE for all the latest Chelsea news .
- Jose Mourinho says Paris Saint-Germain are the most aggressive side they've played.
- Paris Saint-Germain are the most aggresive side his team has played this season.
- Jose Mourinho is the Chelsea boss.
- Chelsea are also called the Blues.
- Paris Saint-Germain are French giants.
- Chelsea hosts Paris Saint-Germain.
- The match is in the Champions League.
- The match is in the last-16 second leg.
- Laurent Blanc claims Chelsea have 'dirty tricks'
- The dirty tricks involve Diego Costa.
- Chelsea have committed more fouls than PSG in the compeition so far.
- David Luiz proved he had a ruthless streak in him.
- This ruthless streak was provied in the last leg.
- The last leg was in Paris.
- Thiago Silva is another danger man.
- Marco Verratti is another danger man.
- Zlatan Ibrahimovic is another danger man.
- CLICK HERE for all the latest Chelsea news .

Please breakdown the following passage into independent facts: Riley Hughes died in a Perth hospital at just 32 days old on March 17 . Parents Greg and Catherine Hughes have set up a Facebook page . They want to stop other parents from having to endure the same heartache . Urged parents to immunise kids to stop preventable childhood deaths . Whooping cough is 'highly infectious' and lethal in babies . Immunisation against it is available for children from two months old . In Australia it is the least well controlled of all vaccine-preventable diseases .
- Riley Hughes died
- Riley Hughes died in a Perth hospital
- Riley Hughes died at just 32 days old
- Riley Hughes died on March 17
- Parents Greg have set up a Facebook page
- Catherine Hughes have set up a Facebook page
- The parent want to stop other parents
- The parent want to stop other parents from having to endure the same heartache
- Urged parents to immunise kids
- Urged parents to stop preventable childhood deaths
- Whooping cough is 'highly infectious'
- Whooping cough is lethal
- Whooping cough is lethal in babies.
- Immunisation against  Whooping cough is available
- Immunisation is available for children from two months old
- Whooping cough is the least well controlled
- Whooping cough is the least well controlled of all vaccine-preventable diseases .
- Whooping cough is the least well controlled in Australia

Please breakdown the following passage into independent facts: Marcin Kostrzewa, 31,  took restricted files from flat next-door . Became 'infatuated' with Shane Spencer after finding out about his work . He contacted Polish embassy and tried to sell the papers for PS50,000 . Jailed for four-and-a-half years after jury finds him guilty of burglary .
- Marcin Kostrzewa is 31.
- Marcin Kostrzewa took restricted files.
- The files were from the flat next-door.
- Marcin Kostrzewa became 'infatuated' with Shane Spencer.
- Marcin Kostrzewa was infatuated after finding out about Shane Spencer's work.
- Marcin Kostrzewa contacted the Polish embassy.
- Marcin Kostrzewa tried to sell the papers.
- The price of the papers was £50,000.
- Marcin Kostrzewa was jailed.
- Marcin Kostrzewa was jailed for four-and-a-half years.
- The jury found Marcin Kostrzewa guilty.
- Marcin Kostrzewa was found guilty of burglary.

Please breakdown the following passage into independent facts: Rare leatherback sea turtle was found stranded on a South Carolina beach . Nicknamed Yawkey, the huge creature was so big he didn't fit on  scales . He is now being treated with fluids and antibiotics at a nearby aquarium . Veterinarians believe he may have become stranded after eating plastic . Sea turtles often mistake plastic debris for jellyfish, their favourite food .
- Sea turtle was found stranded on a beach.
- The turtle was a rare leeatherback tutrtle.
- The beach was in South Carolina.
- The turtle was nicknamed Yawkey.
- The huge creature was so big he didn't fit on scales.

Please breakdown the following passage into independent facts: {Summary}
"""

data = [json.loads(line) for line in open(sys.argv[1])]
out_file = sys.argv[2]

for dat in tqdm(data):
    document = dat["document"]
    summary = dat["summary"]

    message = acu_generation_prompt.format(Summary=summary)
    acus = lm.run_generation(message)

    # print(acus)

    # clean acus
    acus = [xx[2:] for xx in acus.split("\n") if xx.startswith("- ") or (len(xx)>1 and xx[0].isdigit() and xx[1] == ".")]

    dat["acus"] = acus

with open(out_file, "w") as f:
    for dat in data:
        f.write( json.dumps(dat) + "\n" )