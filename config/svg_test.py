import svgwrite

def create_rainbow_triangle():
    # 创建SVG画布
    dwg = svgwrite.Drawing('rainbow_triangle.svg', size=('800px', '800px'))
    
    # 定义三角形的顶点坐标
    points = [(400, 100), (100, 600), (700, 600)]
    
    # 创建三个线性渐变
    gradient1 = dwg.defs.add(dwg.linearGradient(id='gradient1', start=(400, 100), end=(100, 600)))
    gradient2 = dwg.defs.add(dwg.linearGradient(id='gradient2', start=(100, 600), end=(700, 600)))
    gradient3 = dwg.defs.add(dwg.linearGradient(id='gradient3', start=(700, 600), end=(400, 100)))
    
    # 设置渐变颜色
    gradient1.add_stop_color(0, 'rgb(255, 0, 0)', opacity=1)  # 红色
    gradient1.add_stop_color(1, 'rgb(0, 255, 0)', opacity=1)  # 绿色
    
    gradient2.add_stop_color(0, 'rgb(0, 255, 0)', opacity=1)  # 绿色
    gradient2.add_stop_color(1, 'rgb(0, 0, 255)', opacity=1)  # 蓝色
    
    gradient3.add_stop_color(0, 'rgb(0, 0, 255)', opacity=1)  # 蓝色
    gradient3.add_stop_color(1, 'rgb(255, 0, 0)', opacity=1)  # 红色
    
    # 创建三个三角形并应用不同的渐变
    triangle1 = dwg.polygon(points=[(400, 100), (100, 600), (400, 350)], fill='url(#gradient1)')
    triangle2 = dwg.polygon(points=[(100, 600), (700, 600), (400, 350)], fill='url(#gradient2)')
    triangle3 = dwg.polygon(points=[(700, 600), (400, 100), (400, 350)], fill='url(#gradient3)')
    
    # 添加三角形到画布
    dwg.add(triangle1)
    dwg.add(triangle2)
    dwg.add(triangle3)
    
    # 保存SVG文件
    dwg.save()

if __name__ == '__main__':
    create_rainbow_triangle()