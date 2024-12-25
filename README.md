# AI-Chatbot-Python-Flask
An AI Chatbot using Python and Flask REST API 

## Requirements (libraries)
1. TensorFlow
2. NLP
3. Flask
4. Numpy

## VsCode SetUp
1. Clone the repository-> cd into the cloned repository folder
2. Create a python virtual environment 
```
# macOS/Linux
# You may need to run sudo apt-get install python3-venv first
python3 -m venv .venv

# Windows
# You can also use py -3 -m venv .venv
python -m venv .venv
```
When you create a new virtual environment, a prompt will be displayed to allow you to select it for the workspace.

3. Activate the virtual environment
```
#linux
source ./venv/bin/activate  # sh, bash, or zsh

#windows
.\venv\Scripts\activate
```
4. Run ```pip install --upgrade tensorflow``` to install ```Tensorflow```
5. Run ```pip install -U nltk``` to install ```nltk```
6. Run ```pip install -U Flask``` to install ```flask```
7. To expose your bot via Ngrok, run ```pip install flask-ngrok``` to install ```flask-ngrok``` Then you'll need to configure your ngrok credentials(login: email + password) Then uncomment this line ```run_with_ngrok(app) ``` and comment the last two lines ```if __name__ == "__main__": app.run() ``` Notice that ngrok is not used by default.
8. To access your bot on localhost, go to ```http://127.0.0.1:5000/ ``` If you're on Ngrok your url will be ```some-text.ngrok.io```
9. if ngrok isn't exposing the URL correctly automatically, you can manually start ngrok for the Flask app. Here’s how:
  - Open a new terminal window and start ngrok: Run the following command to expose port 5000:
    ``` 
    ngrok http 5000 
    ```


### Execution
To run this Bot: <br>First run the ```train.py``` file to train the model. This will generate a file named ```chatbot_model.keras```<br>
This is the model which will be used by the Flask REST API to easily give feedback without the need to retrain.<br>
After running ```train.py```, next run the ```app.py``` to initialize and start the bot.<br>
To add more terms and vocabulary to the bot, modify the ```intents.json``` file and add your personalized words and retrain the model again.


<!-- Actual text -->
## Find me on
[![Facebook][1.2]][1] [![LinkedIn][2.2]][2] [![Instagram][3.2]][3]

<!-- Icons -->

[1.2]: https://i.imgur.com/dqSkGWu.png (Facebook)
[2.2]: https://raw.githubusercontent.com/MartinHeinz/MartinHeinz/master/linkedin-3-16.png (LinkedIn)
[3.2]: https://i.imgur.com/TFy6wii.png (Instagram)

<!-- Links to my social media accounts -->
[1]: https://facebook.com/fossmentor
[2]: https://www.linkedin.com/in/fossmentor/
[3]: https://www.instagram.com/fossmentor.official/

## Having troubles implementing?
 > Reach out to me contact@fossmentor.com 
 I will be happy to assist 
# 
## want something improved or added?
  > Fork the repo @ [GitHub](https://github.com/fossmentor-official/AI-Chatbot-Python-Flask)
# 
## Regards,
 > [Fossmentor](https://fossmentor.com)