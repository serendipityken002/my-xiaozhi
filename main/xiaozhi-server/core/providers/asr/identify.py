from funasr import AutoModel
import numpy as np
import os
import pickle

def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

class SpeakerDatabase:
    def __init__(self, db_path="speaker_db"):
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
            with open(self.db_path, 'rb') as f:
                self.speakers = pickle.load(f)
    
    def _save_database(self):
        """保存数据库"""
        # 确保目录存在
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.speakers, f)

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
    
    def identify_speaker(self, wav_path, threshold=0.5):
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

def main():
    # 获取单例实例
    speaker_id = SpeakerIdentification()

    # speaker_id.batch_register_speaker("voice_data/雨落倾城")

    speaker, score = speaker_id.identify_speaker("voice_data/雨落倾城/3.wav")
    print(f"识别结果: {speaker}, 相似度得分: {score:.4f}")


if __name__ == "__main__":
    main()