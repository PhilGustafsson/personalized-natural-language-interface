To run the program you will need to install
sounddevice
keyboard

then stand in the record folder and start record.py

record.py will read the entries from the data_script.jsonl 
if the user want to read another file instead this can be changed in
DATA_FILE = "data_script.jsonl"

The script is structued on the form 

{"sentence": 1, "text": "God morgon! Solen har just gått upp över hustaken, och en ny dag börjar."}

the user will get a message to input their username, this will
also be the name of the folder where the users recordings are stored 
recordings/"username"

Each sentence will be a seperate recording, the user will start each
recording by pressing space and stop the recording by pressing space.
Each recording is saved seperately so a user can only read parts of the 
prompts if they choose. To quit early press CTRL-C

The program will only continue based on user input via space. 

