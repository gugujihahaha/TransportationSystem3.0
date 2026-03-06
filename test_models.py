import sys, torch
sys.path.insert(0, '.')

# exp1
from exp1.src.model import TransportationModeClassifier
m1 = TransportationModeClassifier(trajectory_feature_dim=9, num_classes=7, segment_stats_dim=18)
out1 = m1(torch.randn(4, 50, 9), segment_stats=torch.randn(4, 18))
assert out1.shape == (4, 7), f"exp1 shape error: {out1.shape}"
print(f"✅ exp1 OK: {out1.shape}")

# exp4
from exp4.src.model_weather import TransportationModeClassifierWithWeather
m4 = TransportationModeClassifierWithWeather(
    trajectory_feature_dim=9, spatial_feature_dim=15,
    weather_feature_dim=12, segment_stats_dim=18)
out4 = m4(torch.randn(2,50,9), torch.randn(2,50,15),
          torch.randn(2,50,12), segment_stats=torch.randn(2,18))
assert out4.shape == (2, 7), f"exp4 shape error: {out4.shape}"
print(f"✅ exp4 OK: {out4.shape}")

# exp5
from exp5.src.model_weak_supervision import WeaklySupervisedContextModel
m5 = WeaklySupervisedContextModel(segment_stats_dim=18)
out5 = m5(torch.randn(4,50,9), torch.randn(4,50,15),
          torch.randn(4,50,12), segment_stats=torch.randn(4,18))
assert out5.shape == (4, 7), f"exp5 shape error: {out5.shape}"
print(f"✅ exp5 OK: {out5.shape}")

print("\n✅ 全部通过，可以开始训练")
