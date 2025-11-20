import os
import random

from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
import numpy as np
import onnxruntime as ort
import cv2

from astrbot.core.message.components import Plain, Reply
from astrbot.core.star.filter.event_message_type import EventMessageType

device = 'cpu'  # 本人服务器配置的是cpu，若有gpu，可以改为'cuda'
global ort_session


def load_model():
    # 加载 ONNX 模型
    global ort_session
    onnx_model_path = './data/plugins/astrbot_plugin_seiadetect/best_seia_model.onnx'
    ort_session = ort.InferenceSession(onnx_model_path)


# 图像预处理（使用 OpenCV）
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))  # 根据模型要求调整图像大小
    image = image.astype(np.float32) / 255.0  # 将像素值归一化到 [0, 1] 范围

    # 使用 ImageNet 数据集的均值和标准差进行标准化
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std  # 标准化

    # 转换为 (C, H, W) 格式，并增加 batch 维度
    image = np.transpose(image, (2, 0, 1))  # 转换为 (C, H, W) 格式
    image = np.expand_dims(image, axis=0)  # 增加 batch 维度

    return image


def preprocess_image_segmentation(image_path, row, col):
    try:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (row * 224, col * 224))  # 根据模型要求调整图像大小
        image = image.astype(np.float32) / 255.0  # 将像素值归一化到 [0, 1] 范围

        # 使用 ImageNet 数据集的均值和标准差进行标准化
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std  # 标准化

        # 转换为 (C, H, W) 格式，并增加 batch 维度
        image = np.transpose(image, (2, 0, 1))  # 转换为 (C, H, W) 格式

        segmented_images = []
        for i in range(row):
            for j in range(col):
                # 计算每个小图像的起始位置
                start_x = i * 224
                start_y = j * 224
                # 切割出224x224的小图像
                segmented_image = image[:, start_y:start_y + 224, start_x:start_x + 224]
                segmented_image = np.expand_dims(segmented_image, axis=0)
                # 将切割的小图像添加到列表中
                segmented_images.append(segmented_image)

        # 返回分割后的图像列表
        return True, segmented_images
    except Exception as e:
        print(e)
        return False, []


def predict_image_segmentation(image_list):
    try:
        results = []
        input_name = ort_session.get_inputs()[0].name
        for image in image_list:
            # print(image.shape)
            ort_inputs = {input_name: image}
            ort_outs = ort_session.run(None, ort_inputs)

            # 使用 argmax 获取最大值所在的位置
            pred = np.argmax(ort_outs[0], axis=1)

            if pred.item() == 1:
                results.append((True, ort_outs[0][0][1]))  # 如果预测为1，记录结果
            else:
                results.append((False, 0.0))  # 否则返回 False

        # 返回所有预测结果
        return True, results
    except Exception as e:
        print(e)
        return False, []


def get_max_prediction(results):
    # 从 results 中筛选出预测为 True 的项
    true_results = [result for result in results if result[0] is True]
    # 如果有 True 结果，找到最大的 ort_outs[0][1]
    if true_results:
        max_value = max(true_results, key=lambda x: x[1])[1]  # 获取最大值
        return True, max_value
    else:
        return False, 0.0


def predict_image(image_path):
    # 预处理图像
    image = preprocess_image(image_path)

    # 获取模型输入名称
    input_name = ort_session.get_inputs()[0].name

    # 推理：获取预测结果
    ort_inputs = {input_name: image}
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs)
    # 使用 argmax 获取最大值所在的位置
    pred = np.argmax(ort_outs[0], axis=1)
    return pred.item() == 1, ort_outs[0][0][1]


@register("SeiaDetect", "orchidsziyou", "检测发送的图片是不是Seia", "1.0.0")
class MyPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        load_model()

    @filter.event_message_type(EventMessageType.ALL)
    async def detect_seia(self, event: AstrMessageEvent):
        # obj=event.message_obj
        message_chain = event.get_messages()
        if len(message_chain) == 1:
            # print(message_chain[0])
            if message_chain[0].type == "Image":
                image_filename = message_chain[0].file
                image_url = message_chain[0].url
                if "/club/item/" in image_url:
                    print("跳过qq官方表情")
                    return
                from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import \
                    AiocqhttpMessageEvent
                assert isinstance(event, AiocqhttpMessageEvent)
                client = event.bot
                payloads2 = {
                    "file_id": image_filename
                }
                response = await client.api.call_action('get_image', **payloads2)
                # print(response)
                image_path = response['file']
                str_path = str(image_path)
                if str_path.endswith('.gif'):
                    return
                if "Emoji" in str_path:
                    return
                if os.path.exists(image_path):
                    try:
                        result, prob = predict_image(image_path)
                        print(result)
                        if result:
                            if prob > 0.7:  # 可以修改阈值
                                message_chain = [
                                    Reply(id=event.message_obj.message_id),
                                    Plain("老婆")
                                ]
                            else:
                                message_chain = [
                                    Reply(id=event.message_obj.message_id),
                                    Plain("可能是我老婆")
                                ]

                            yield event.chain_result(message_chain)
                        else:
                            return
                    except:
                        print("error")
                    else:
                        pass
                else:
                    pass

    @filter.command("searchseia", "搜索老婆")
    async def search_seia(self, event: AstrMessageEvent, row: int = 2, col: int = 2):
        ''' 这是一个 searchseia 命令，用于搜索图片是否包括圣娅'''

        if row <= 0 or col <= 0:
            yield event.plain_result("参数错误")
            return
        if row * col > 16:
            yield event.plain_result("图片分的太多啦，服务器性能爆炸了！")
            return

        # 获取图片原地址
        event.should_call_llm(False)
        message_chain = event.get_messages()

        for msg in message_chain:
            if msg.type == 'Reply':
                # 处理回复消息
                from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
                assert isinstance(event, AiocqhttpMessageEvent)
                client = event.bot
                payload = {
                    "message_id": msg.id
                }
                response = await client.api.call_action('get_msg', **payload)  # 调用 协议端  API
                reply_msg = response['message']
                # print(reply_msg)

                reply_id = msg.id

                for msg in reply_msg:
                    # print(msg)
                    if msg['type'] == 'image':
                        # 官方表情没办法保存
                        picture_url = msg['data']['url']
                        # print(picture_url)
                        if "/club/item/" in picture_url:
                            yield event.plain_result("跳过qq官方表情")
                            return

                        from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import \
                            AiocqhttpMessageEvent
                        assert isinstance(event, AiocqhttpMessageEvent)
                        client = event.bot
                        payloads2 = {
                            "file_id": msg['data']['file']
                        }
                        response = await client.api.call_action('get_image', **payloads2)  # 调用 协议端  API
                        localdiskpath = response['file']

                        # 图片原地址
                        abs_imgpath = os.path.abspath(localdiskpath)
                        print(abs_imgpath)

                        # 切割图片
                        Check, segmented_images = preprocess_image_segmentation(abs_imgpath, row, col)
                        if not Check:
                            yield event.plain_result("切割图片失败")
                            return

                        # 预测图片
                        Check, result_list = predict_image_segmentation(segmented_images)
                        if not Check:
                            yield event.plain_result("预测图片失败")
                            return
                        result, prob = get_max_prediction(result_list)
                        print(str(row * col) + "张图片预测结果：" + str(result) + ", " + str(prob))
                        if result:
                            if prob > 0.7:  # 可以修改阈值
                                message_chain = [
                                    Reply(id=reply_id),
                                    Plain("老婆")
                                ]
                            else:
                                message_chain = [
                                    Reply(id=reply_id),
                                    Plain("可能是我老婆")
                                ]

                            yield event.chain_result(message_chain)
                            return
                        else:
                            # 增加一个小概率出现其他的回复
                            if random.random() < 0.1:
                                message_chain = [
                                    Reply(id=reply_id),
                                    Plain("虽然没找到圣娅，但这也是我老婆")
                                ]
                            else:

                                message_chain = [
                                    Reply(id=reply_id),
                                    Plain("没找到我老婆")
                                ]
                            yield event.chain_result(message_chain)
                            return
