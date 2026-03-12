import os
import shutil
import unittest
from PIL import Image

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.deduplicator.ray_bts_image_minhash_deduplicator import (
    RayImageBTSMinhashDeduplicator,
    RayImageBTSMinhashDeduplicatorWithUid,
)
from data_juicer.utils.constant import HashKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG


class RayImageBTSMinhashDeduplicatorTest(DataJuicerTestCaseBase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # 创建临时目录用于存放测试图片
        cls.img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images_tmp")
        if not os.path.exists(cls.img_dir):
            os.makedirs(cls.img_dir)

        # 生成测试用的简单图片
        cls.img_paths = {
            "red1": os.path.join(cls.img_dir, "red1.jpg"),
            "red2": os.path.join(cls.img_dir, "red2.jpg"),  # 预期与 red1 重复
            "blue1": os.path.join(cls.img_dir, "blue1.jpg"),
            "green1": os.path.join(cls.img_dir, "green1.jpg"),
            "green2": os.path.join(cls.img_dir, "green2.jpg"),  # 预期与 green1 重复
        }

        Image.new("RGB", (224, 224), color="red").save(cls.img_paths["red1"])
        Image.new("RGB", (224, 224), color="red").save(cls.img_paths["red2"])
        Image.new("RGB", (224, 224), color="blue").save(cls.img_paths["blue1"])
        Image.new("RGB", (224, 224), color="green").save(cls.img_paths["green1"])
        Image.new("RGB", (224, 224), color="green").save(cls.img_paths["green2"])

    @classmethod
    def tearDownClass(cls):
        # 清理临时图片和目录
        if os.path.exists(cls.img_dir):
            shutil.rmtree(cls.img_dir)
        super().tearDownClass()

    def _run_image_minhash_dedup(self, dataset: Dataset, target_count, op):
        # 运行算子进行去重
        res_dataset = op.run(dataset)
        res_list = res_dataset.to_pydict()[op.image_key]
        self.assertEqual(len(res_list), target_count)

    @TEST_TAG("ray")
    def test_image_deduplication(self):
        # 准备数据集：包含 5 张图片，其中有 2 组重复（红、绿各两张），1 张独立（蓝）
        ds_list = [
            {"images": [self.img_paths["red1"]]},
            {"images": [self.img_paths["red2"]]},
            {"images": [self.img_paths["blue1"]]},
            {"images": [self.img_paths["green1"]]},
            {"images": [self.img_paths["green2"]]},
        ]

        # 去重后预期应该只剩 3 张独特的图片
        target_count = 3

        # 测试普通算子
        dataset = self.generate_dataset(ds_list)
        work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image_dedup_tmp")

        # 考虑到模型可能对纯色图片的特征提取差异，这里将阈值设高一点，或者直接依赖完全相同的像素
        op = RayImageBTSMinhashDeduplicator(jaccard_threshold=0.9, work_dir=work_dir)
        self._run_image_minhash_dedup(dataset, target_count, op)

        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)

    @TEST_TAG("ray")
    def test_image_deduplication_with_uid(self):
        ds_list = [
            {"images": [self.img_paths["red1"]]},
            {"images": [self.img_paths["red2"]]},
            {"images": [self.img_paths["blue1"]]},
            {"images": [self.img_paths["green1"]]},
            {"images": [self.img_paths["green2"]]},
        ]

        # 为数据集手动添加 uid
        for i, ds in enumerate(ds_list):
            ds[HashKeys.uid] = i

        dataset = self.generate_dataset(ds_list)
        target_count = 3

        work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image_dedup_uid_tmp")
        op = RayImageBTSMinhashDeduplicatorWithUid(jaccard_threshold=0.9, work_dir=work_dir)
        self._run_image_minhash_dedup(dataset, target_count, op)

        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    unittest.main()
