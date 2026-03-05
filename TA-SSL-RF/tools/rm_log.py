import os


def delete_small_logs(folder_path: str):
    a = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                f.close()
            if len(lines) < 200:
                try :
                    os.remove(file_path)
                except:
                    continue
                print(f'{a} Deleted {file_path}\n')
                a = 1+a


folder_path = r'D:\gw\GW-main\SVM\out\log'  # 替换为您的文件夹路径
delete_small_logs(folder_path)
