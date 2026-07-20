"""工具注册中心: 统一以JSON Schema描述工具, 供Agent(LLM或规则规划器)调用."""

import numpy as np

from . import analysis, denoise, enhance, metrics

TOOL_SCHEMAS = [
    {
        "name": "analyze_image",
        "description": "对医学影像进行感知分析, 返回噪声强度sigma、噪声等级、模态、边缘密度、纹理复杂度等",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "denoise_image",
        "description": "使用指定算法对图像去噪, 可选算法: gaussian/median/nlm/wavelet/bm3d/sa-bm3d",
        "input_schema": {
            "type": "object",
            "properties": {
                "method": {"type": "string", "enum": list(denoise.DENOISERS)},
                "sigma": {"type": "number", "description": "噪声标准差(0-255尺度)"},
            },
            "required": ["method"],
        },
    },
    {
        "name": "enhance_image",
        "description": "对图像进行质量增强, 可选方法: clahe(对比度)/unsharp(锐化)/gamma(亮度)",
        "input_schema": {
            "type": "object",
            "properties": {
                "method": {"type": "string", "enum": list(enhance.ENHANCERS)},
            },
            "required": ["method"],
        },
    },
    {
        "name": "evaluate_quality",
        "description": "评估当前图像质量: 有参考图时返回PSNR/SSIM/RMSE, 同时返回无参考指标",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


class ToolExecutor:
    """持有图像状态并执行工具调用. Agent每次调用工具都会更新工作图像."""

    def __init__(self, image: np.ndarray, reference: np.ndarray | None = None,
                 filename: str = ""):
        self.original = image.copy()
        self.current = image.copy()
        self.denoised: np.ndarray | None = None
        self.reference = reference
        self.filename = filename
        self.trace: list[dict] = []

    def execute(self, name: str, args: dict) -> dict:
        if name == "analyze_image":
            result = analysis.analyze_image(self.current, self.filename)
        elif name == "denoise_image":
            sigma = args.get("sigma") or analysis.estimate_noise_sigma(self.current)
            self.current = denoise.denoise(self.current, args["method"], sigma)
            self.denoised = self.current.copy()
            result = {"status": "ok", "method": args["method"], "sigma": round(sigma, 2)}
        elif name == "enhance_image":
            self.current = enhance.enhance(self.current, args["method"])
            result = {"status": "ok", "method": args["method"]}
        elif name == "evaluate_quality":
            # 无参考指标评估最终增强结果; 全参考指标评估去噪保真度
            # (对比度增强会刻意改变灰度分布, 不适合用PSNR/SSIM衡量)
            result = metrics.no_reference_metrics(self.current)
            if self.reference is not None:
                target = self.denoised if self.denoised is not None else self.current
                result.update(metrics.full_reference_metrics(self.reference, target))
        else:
            result = {"error": f"未知工具: {name}"}
        self.trace.append({"tool": name, "args": args, "result": result})
        return result
