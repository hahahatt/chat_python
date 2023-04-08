import threading
import json

from config.DatabaseConfig import *
from utils.Database import Database
from utils.BotServer import BotServer
from utils.Preprocess import Preprocess
from models.intent.IntentModel import IntentModel
from models.ner.NerModel import NerModel
from utils.FindAnswer import FindAnswer

p=Preprocess(word2index_dic='train_tools/dict/chatbot_dict.bin',userdic='utils/user_dic.tsv')

intent=IntentModel(model_name='models/intent/intent_model.h5',proprocess=p)

ner=NerModel(model_name='models/ner/ner_model.h5',proprocess=p)

def to_client(conn,addr,params):
    db=params['db']
    try:
        db.connect()

        read=conn.recv(2048)
        print("==========================")
        print('Connection from : %s'%str(addr))

        if read is None or not read:
            print('클라이언트 연결 끊어짐')
            exit(0)

        recv_json_data=json.loads(read.decode())
        print("데이터 수신 : ",recv_json_data)
        query=recv_json_data['Query']

        intent_predict=intent.predict_class(query)
        intent_name=intent.labels[intent_predict]

        ner_predicts=ner.predict(query)
        ner_tags=ner.predict_tags(query)

        try:
            f=FindAnswer(db)
            answer_text, answer_image=f.search(intent_name,ner_tags)
            answer=f.tag_to_word(ner_predicts,answer_text)

        except:
            answer="죄송 무슨 말인지 잘 모르겠어여,,, 공부해봄.."
            answer_image=None

        send_json_data_str={
            "Query" : query,
            "Answer": answer,
            "AnswerImageUrl": answer_image,
            "Intent": intent_name,
            "NER": str(ner_predicts)
        }
        message=json.dumps(send_json_data_str).encode()
        conn.send(message)

    except Exception as ex:
        print(ex)

    finally:
        if db is not None:
            db.close()
        conn.close()

if __name__=='__main__':
    db=Database(
        host=DB_HOST, user=DB_USER,password=DB_PASSWORD, db_name=DB_NAME
    )
    print("DB 접속")

    port=5050
    listen=100
    bot=BotServer(port,listen)
    bot.create_sock()
    print("bot start")

    while True:
        conn,addr=bot.ready_for_client()
        params={
            "db":db
        }
        client=threading.Thread(target=to_client,args=(
            conn,
            addr,
            params
        ))
        client.start()






