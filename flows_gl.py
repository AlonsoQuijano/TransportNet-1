import numpy as np
from glumpy import app, glm, gloo, gl
from glumpy.graphics.collections import SegmentCollection, MarkerCollection
from glumpy.transforms import Position, OrthographicProjection, PanZoom, Viewport
from numpy.random import random
import pandas as pd

links = pd.read_csv('data/flows_to_draw.csv')
links.flow /= links.flow.max()
# links = links[::10]
dx, dy = links.xa.mean(), links.ya.mean()
links.xa -= dx
links.xb -= dx
links.ya -= dy
links.yb -= dy


ss = pd.read_csv('data/soursinks.csv')
ss.x -= dx
ss.y -= dy

# app.use('freeglut')
window = app.Window(width=1200, height=600, color=(1,1,1,1))

@window.event
def on_draw(dt):
    window.clear()
    segments.draw()
    markers.draw()

@window.event
def on_key_press(key, modifiers):
    if key == app.window.key.SPACE:
        transform.reset()

n = 100
P0 = np.dstack((np.linspace(100,1100,n),np.ones(n)* 50,np.zeros(n))).reshape(n,3)
P1 = np.dstack((np.linspace(110,1110,n),np.ones(n)*550,np.zeros(n))).reshape(n,3)
viewport = Viewport()
transform = PanZoom(OrthographicProjection(Position()))
segments = SegmentCollection(mode="agg", linewidth='local', transform=transform, viewport=viewport)
# segments.append([[0,0,0], [1, 0, 0]],[[1,100.112,0], [1000.11, 100.234234, 0]] ,
#                 linewidth = [2,3], color=[[0, 0, 1, 1]])

for i in links.index:
    f = links.flow[i]
    I = 0.3 + f / 2 if f > 0 else 0.05
    r = f
    segments.append(np.array([links.xa[i], links.ya[i], 0]), np.array([links.xb[i], links.yb[i], 0]),
                    linewidth=3, color = (r, 0, 0, I))

N = np.random.uniform(0, 800, (n,3)) * (1,1,0)
markers = MarkerCollection(marker='disc', transform=transform, viewport=viewport)
# markers.append(N, size=15, linewidth=2, itemsize=1,
#                fg_color=(0,0,0,1), bg_color=(0,0,0,1))
markers.append(np.array([ss.x, ss.y, np.zeros(len(ss.x))]).T, size=10, linewidth=2,
               fg_color=(1,1,0,1), bg_color=(1,.5,.5,1))

segments['antialias'] = 1

window.attach(transform)
window.attach(viewport)
app.run()
