from funasr import AutoModel
import numpy as np
import os
import json
import time

def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

class SpeakerDatabase:
    def __init__(self, db_path="speaker_db.json"):
        """初始化说话人数据库
        
        Args:
            db_path: 数据库文件路径
        """
        # 使用绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(current_dir, db_path)
        self.speakers = {}  # {speaker_name: [embeddings]}
        self._load_database()
    
    def _load_database(self):
        """加载数据库"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 将列表转换回numpy数组
                self.speakers = {}
                for speaker, embeddings_list in data.items():
                    self.speakers[speaker] = [np.array(emb) for emb in embeddings_list]
                print(f"已加载声纹数据库，共有 {len(self.speakers)} 个说话人")
            except Exception as e:
                print(f"加载声纹数据库失败: {e}")
                self.speakers = {}
    
    def _save_database(self):
        """保存数据库"""
        # 确保目录存在
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # 将numpy数组转换为列表以便JSON序列化
        data_to_save = {}
        for speaker, embeddings in self.speakers.items():
            data_to_save[speaker] = [emb.tolist() for emb in embeddings]
        
        # 保存为JSON格式
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False)
        
        # 同时保存一个可读的信息文件
        info_path = os.path.splitext(self.db_path)[0] + "_info.txt"
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(f"声纹数据库信息 - 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for speaker, embeddings in self.speakers.items():
                f.write(f"说话人: {speaker}\n")
                f.write(f"  注册声纹数量: {len(embeddings)}\n")
                for i, emb in enumerate(embeddings):
                    f.write(f"  声纹 #{i+1}: 维度 {emb.shape}, 均值 {np.mean(emb):.4f}, 方差 {np.var(emb):.4f}\n")
                f.write("\n")

class SpeakerIdentification:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 初始化模型和数据库
            cls._instance.model = AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common")
            cls._instance.db = SpeakerDatabase()
        return cls._instance
    
    def register_speaker(self, wav_path, speaker_name):
        """注册说话人声纹"""
        results = self.model.generate(input=[wav_path])
        embedding = results[0]['spk_embedding']
        if embedding.is_cuda:
            embedding = embedding.cpu().numpy().squeeze()
        else:
            embedding = embedding.numpy().squeeze()
        
        if speaker_name not in self.db.speakers:
            self.db.speakers[speaker_name] = []
        self.db.speakers[speaker_name].append(embedding)
        
        self.db._save_database()
        print(f"已将说话人 {speaker_name} 的声纹特征注册到数据库")
    
    def identify_speaker(self, wav_path, threshold=0.55):
        """识别说话人身份"""
        if not self.db.speakers:
            return "数据库为空", 0.0
        
        results = self.model.generate(input=[wav_path])
        test_embedding = results[0]['spk_embedding']
        if test_embedding.is_cuda:
            test_embedding = test_embedding.cpu().numpy().squeeze()
        else:
            test_embedding = test_embedding.numpy().squeeze()
        
        max_score = -1
        identified_speaker = None
        
        for speaker, embeddings in self.db.speakers.items():
            scores = [cosine_similarity(test_embedding, emb) for emb in embeddings]
            avg_score = np.mean(scores)
            
            if avg_score > max_score:
                max_score = avg_score
                identified_speaker = speaker
        
        if max_score >= threshold:
            return identified_speaker, max_score
        else:
            return "未知说话人", max_score
    
    def batch_register_speaker(self, wav_dir):
        """批量注册说话人"""
        for file in os.listdir(wav_dir):
            if file.endswith('.wav'):
                wav_path = os.path.join(wav_dir, file)
                speaker_name = wav_dir.split('/')[-1]
                self.register_speaker(wav_path, speaker_name)
        print(f"已将目录 {wav_dir} 中的所有说话人声纹特征注册到数据库")

# def main():
#     # 获取单例实例
#     speaker_id = SpeakerIdentification()

#     # speaker_id.batch_register_speaker("voice_data/雨落倾城")

#     speaker, score = speaker_id.identify_speaker("voice_data/雨落倾城/3.wav")
#     print(f"识别结果: {speaker}, 相似度得分: {score:.4f}")


# if __name__ == "__main__":
#     main()