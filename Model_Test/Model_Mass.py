# 计算xml文件

import xml.etree.ElementTree as ET

def calculate_total_mass(xml_file):
    # 解析 XML 文件
    tree = ET.parse(xml_file)
    root = tree.getroot()

    total_mass = 0.0

    # 遍历所有的 body 元素
    for body in root.iter('body'):
        # 获取 body 的 name 属性
        body_name = body.attrib.get('name')

        # 查找 body 下的 inertial 元素，并获取其 mass 属性
        inertial = body.find('inertial')
        if inertial is not None:
            mass = inertial.attrib.get('mass')
            if mass:
                print(f"Body: {body_name}, Mass: {mass} kg")
                total_mass += float(mass)

    return total_mass

# 使用示例
xml_file = r"D:\Code\Model\Sengi_simple_single\Sengi_simple_single.xml" # 替换为你的 XML 文件路径
total_mass = calculate_total_mass(xml_file)
print(f"Total mass of all parts: {total_mass} kg")
