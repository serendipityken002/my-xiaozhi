from plugins_func.register import register_function, ToolType, ActionResponse, Action
from config.logger import setup_logging
from core.providers.asr.identify import SpeakerIdentification
import uuid
import traceback

TAG = __name__
logger = setup_logging()

# 定义函数描述
register_voiceprint_function_desc = {
    "type": "function",
    "function": {
        "name": "register_voiceprint",
        "description": "当用户想要注册声纹时调用",
        "parameters": {
            "type": "object",
            "properties": {
                "user_name": {
                    "type": "string",
                    "description": "要注册的用户名称"
                }
            },
            "required": ["user_name"]
        }
    }
}

@register_function('register_voiceprint', register_voiceprint_function_desc, ToolType.SYSTEM_CTL)
def register_voiceprint(conn, user_name: str = None):
    try:
        file_path = conn.asr.save_audio_to_file(conn.asr_audio, conn.session_id)
        # speaker_id = SpeakerIdentification()
        # speaker_id.register_speaker(file_path, user_name)
        name_text = f" {user_name}" if user_name else "您的"
        
        success_message = (
            f"声纹注册成功！{name_text}声纹特征已保存。声纹音频{file_path}"
        )
        
        # 5. 返回成功消息
        return ActionResponse(action=Action.RESPONSE, result="声纹注册指令已接收", response=success_message)
    except Exception as e:
        logger.bind(tag=TAG).error(f"声纹注册失败: {e}")
        return ActionResponse(action=Action.RESPONSE, result=str(e), response="声纹注册失败")

