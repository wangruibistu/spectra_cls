import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


wavelengths = np.linspace(3500, 9500, 1000)  # 假设在3500到9500 Å范围内
spectrum = np.sin(wavelengths / 500)  # 示例输入光谱

templates = {
    'galaxy': np.sin(wavelengths / 510),
    'star_O': np.cos(wavelengths / 480),
    'star_B': np.cos(wavelengths / 500),
    'star_A': np.sin(wavelengths / 490),
    'star_F': np.sin(wavelengths / 470),
    'star_G': np.cos(wavelengths / 510),
    'star_K': np.sin(wavelengths / 520),
    'star_M': np.cos(wavelengths / 530),
    'quasar': np.sin(wavelengths / 450)
}

def preprocess_spectrum(wavelengths, spectrum, template_wavelengths):
    # 插值到相同的波长范围和分辨率
    interpolator = interp1d(wavelengths, spectrum, kind='linear', bounds_error=False, fill_value="extrapolate")
    resampled_spectrum = interpolator(template_wavelengths)
    return resampled_spectrum

def redshift_spectrum(wavelengths, spectrum, z):
    # 对光谱进行红移
    redshifted_wavelengths = wavelengths * (1 + z)
    return redshifted_wavelengths

def fit_continuum(wavelengths, spectrum, degree=3):
    # 拟合多项式连续谱
    coeffs = np.polyfit(wavelengths, spectrum, degree)
    continuum = np.polyval(coeffs, wavelengths)
    return continuum

def chi_squared(observed, expected, obs_err=None):
    if obs_err is None:
        obs_err = np.ones_like(observed)*0.01
    # 计算卡方值
    return np.mean((observed - expected) ** 2 / obs_err**2)

def match_spectrum(wavelengths, spectrum, templates):
    best_match = None
    lowest_chi2 = np.inf

    for label, template in templates.items():
        for z in np.linspace(0, 0.1, 100):  # 假设红移在0到0.1之间
            redshifted_template_wavelengths = redshift_spectrum(wavelengths, wavelengths, z)
            redshifted_template = preprocess_spectrum(redshifted_template_wavelengths, template, wavelengths)
            
            # 校正连续谱
            continuum = fit_continuum(wavelengths, redshifted_template)
            corrected_template = redshifted_template / continuum
            
            continuum = fit_continuum(wavelengths, spectrum)
            corrected_spectrum = spectrum / continuum
            
            chi2 = chi_squared(corrected_spectrum, corrected_template, None)

            if chi2 < lowest_chi2:
                lowest_chi2 = chi2
                best_match = (label, z)

    return best_match, lowest_chi2

# 预处理输入光谱和模板
template_wavelengths = wavelengths  # 假设模板波长范围与输入相同
processed_spectrum = preprocess_spectrum(wavelengths, spectrum, template_wavelengths)

# 进行模板匹配
best_match, chi2 = match_spectrum(wavelengths, processed_spectrum, templates)

print(f"Best match: {best_match[0]} with redshift: {best_match[1]:.4f} and chi-squared: {chi2}")

# 可视化匹配结果
plt.plot(wavelengths, processed_spectrum, label='Input Spectrum')
best_template = templates[best_match[0]]
redshifted_wavelengths = redshift_spectrum(wavelengths, wavelengths, best_match[1])
redshifted_template = preprocess_spectrum(redshifted_wavelengths, best_template, wavelengths)
continuum = fit_continuum(wavelengths, redshifted_template)
corrected_template = redshifted_template / continuum
continuum = fit_continuum(wavelengths, spectrum)
corrected_spectrum = spectrum / continuum
plt.plot(wavelengths, 
         corrected_template, 
         label=f'Template ({best_match[0]})', 
         linestyle='dashed'
         )
plt.xlabel('Wavelength (Å)')
plt.ylabel('Intensity')
plt.legend()
plt.show()
