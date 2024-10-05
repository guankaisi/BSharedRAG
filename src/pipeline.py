from bsharedrag import BSharedRAG
import argparse
import time


def interact_with_user(model):
    print("你好！我可以帮你回答问题。输入你的问题，或者Ctrl+C退出。")
    while True:
        try:
            user_input = input("请输入你的问题: ")
            # 因为模型期待一个问题列表，我们把用户输入放在一个列表中
            start = time.time()
            # questions = [f"query:{user_input}</s>"]
            # 调用pipeline方法来处理问题并获取回答
            answers = model.pipeline([user_input])
            # 打印回答。因为返回的是一个列表，我们取列表的第一个元素
            print("\n\n回答:", answers[0])
            end = time.time()
            print("Time of RAG : {}".format(end - start))
        except KeyboardInterrupt:
            print("\n再见！")
            break
        # except Exception as e:
        #     print("处理你的问题时出错了。")
        #     exit()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='bshared', help='bshared or bge')
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--base_model_path', type=str, default='/data4/kaisi/bsharedrag/model/domain_model')
    parser.add_argument('--emb_peft_path',type=str,default='/data4/kaisi/bsharedrag/model/model_replbaichuan_domain_all_ecom_plus')
    parser.add_argument('--gen_peft_path',type=str,default='/data4/kaisi/bsharedrag/model/Baichuan2-7B-Domain_v3_belle_20w_ecom_retrieve')
    parser.add_argument('--dense_index',type=str,default='/data4/kaisi/bsharedrag/index/baichuan_title_domain.index')
    parser.add_argument('--database_path',type=str,default='/data4/kaisi/bsharedrag/database/database_update.json')
    args = parser.parse_args()
    bsharedrag = BSharedRAG(args)
    interact_with_user(bsharedrag)
    
            
    