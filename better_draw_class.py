#The core version of this file lives in Misc notebooks/Image stuff/better_draw_class.py
#until such time as I figure out how packages work 
from PIL import Image, ImageDraw, ImageFont
import IPython.display
import imageio
import numpy as np
import math
import svgwrite
show = IPython.display.display

def blank_image(*dimensions):
    return Image.fromarray(np.ones(list(dimensions)+[3],dtype=np.uint8)*255)

#first, define a general structure that all draw classes should inherit from
class Drawer:
    """
    This is the base class for the low-level drawer classes
    that implement a particular image library's functionality for
    drawing basic shapes on an image. We want to implement:
    * Line(p, q, color)
    * Circle(p, r, color, fill)
    * Polygon(pts, color, fill)
    * Text(xy, text, color, font_size, font=default_font)
    Any of these can also take custom kwargs.
    """
    pass


class PIL_drawer:
    def __init__(self,draw):
        self.draw = draw
    def line(self,p,q,color=''):
        self.draw.line(list(p)+list(q),fill=color)
    def circle(self,p,r,fill='',outline=''):
        self.draw.ellipse([p[0]-r,p[1]-r,p[0]+r,p[1]+r],fill=fill,outline=outline)
    def poly(self,pts,fill='',outline=''):
        self.draw.polygon(pts,fill=fill,outline=outline)
    def text(self,xy,text,color='',font_size = 20, actual_font = None,anchor='mm'):
        if actual_font is None:
            actual_font = ImageFont.truetype("/Users/Drake/Downloads/font.ttf", font_size)
        self.draw.text(xy,text,font=actual_font,fill=color,anchor=anchor)
    def text_box(self,xy,text,color='',font_size = 20, actual_font = None,anchor='mm',border=0.1):
        if actual_font is None:
            actual_font = ImageFont.truetype("/Users/Drake/Downloads/font.ttf", font_size)
        a,b,c,d = self.draw.textbbox(xy, text=text, font=actual_font,anchor=anchor)
        #hack to make boxes not be so tall
        a,b,c,d = [x*0.5 for x in [a,b,c,d]]
        border *= font_size
        a,b,c,d = a-border,b-border,c+border,d+border
        self.draw.polygon([(a,b),(c,b),(c,d),(a,d)],outline='black',fill='white')
        self.text(xy,text,color,font_size,actual_font,anchor)

#for interfacing with the svgwrite library
#TODO: finish
def svg_color(color):
    if color=='' or color is None:
        return 'none'
    if type(color)==str:
        return color
    elif type(color) in [tuple,list,np.ndarray]:
        return svgwrite.rgb(color[0],color[1],color[2], mode='RGB')
class svgwrite_drawer:
    def __init__(self,filename,dims,debug=False):
        self.draw = svgwrite.Drawing(filename=filename,size=dims)
    def line(self,p,q,color='black',stroke_width=1):
        self.draw.add(self.draw.line(
            start=p,end=q,stroke=svg_color(color),
            stroke_width=stroke_width
            ))
    def circle(self,p,r,fill='none',outline='black',opacity=1):
        circle = self.draw.circle(center=p,r=r,stroke=svg_color(outline))
        circle.fill(svg_color(fill), opacity=opacity)
        self.draw.add(circle)
    def ellipse(self,p,r,fill='none',outline='black',opacity=1):
        pass
    def poly(self,pts,fill='none',outline='black',opacity=1,stroke_width=1):
        poly = self.draw.polygon(points=pts,stroke=svg_color(outline),stroke_width=stroke_width)
        poly.fill(svg_color(fill), opacity=opacity)
        self.draw.add(poly)
    def text(self,xy,text,color='black',font_size = 20, actual_font = None,anchor='mm', italics=False, bold=False,outline=None,fill=None):
        #very hacky, discards a bunch of the arguments fed to it by the parent class
        if font is None:
            font = "Arial"
        text_style = f"""font-size:{font_size};
        {'font-style: italic;' if italics else ''}
        {'font-weight: bold;' if bold else ''}
        {'stroke: '+svg_color(outline) if outline is not None else ''}
        """
        self.draw.add(self.draw.text(
            text, insert=xy, fill=svg_color(color), style=text_style
            ))
    def save(self):
        self.draw.save() 



#Improved Draw class that takes different base draw objects
class Draw:
    def __init__(self,img,center=True,y_up = True, units='deg', draw_type='imagedraw', filename='untitled.svg'):
        self.draw_type = draw_type.lower()
        if type(img)==tuple:
            #we're only creating this image for the draw class, so we'll keep it around in a field
            if self.draw_type == 'imagedraw':
                img = blank_image(*img)
                self.img = img
                self.dim = img.size
            elif self.draw_type == 'svg':
                self.dim = img
                self.img = None
        else:
            self.img = img
            self.dim = img.size
        if self.draw_type == 'imagedraw':
            self.draw = PIL_drawer(ImageDraw.Draw(img))
        elif self.draw_type == 'svg':
            self.draw = svgwrite_drawer(filename=filename, dims=self.dim)
        else:
            raise Exception("Unknown draw type: "+self.draw_type)
        self.abs_scale = 1
        self.abs_shift = [0,0]
        self.center = center
        self.r = 0
        self.original_y_up = y_up
        self.y_up = y_up
        self.units=units
        self.default_color = 'black'
        self.default_fill = 'white'
        self.labels = {}
        self.checkpoints = {}
        if self.center:
            self.shift(self.dim[0]/2,self.dim[1]/2)
    def checkpoint(self,key):
        #we need to copy abs_shift or it'll get modified
        value = (self.abs_scale,self.abs_shift[0],self.abs_shift[1],self.r,self.y_up)
        self.checkpoints[key] = value
    def return_to(self,checkpoint):
        #we can only return to a checkpoint if it was set at some point in the past
        assert checkpoint in self.checkpoints, checkpoint
        sc,sh1,sh2,r,y_up = self.checkpoints[checkpoint]
        self.abs_scale = sc
        self.abs_shift = [sh1,sh2]
        self.r = r
        self.y_up = y_up
    def scale(self,s):
        #Scale about the current origin, which is at xshift,yshift
        self.abs_scale *= s
        self.abs_shift = [self.abs_shift[0]/s, self.abs_shift[1]/s]
    def shift(self,s,second_coord=None):
        if second_coord is not None:
            s = [s,second_coord]
        true_shift = self.point_rotate(s)
        self.abs_shift[0]+=true_shift[0]
        self.abs_shift[1]+=true_shift[1]
    def reflect_across(self,p):
        """Reflect across the line from the origin to p. 
        TODO: make this reflect across the line from p to q."""
        #we'll have to toggle y_up and then make the rotation 
        self.y_up = not self.y_up
        theta = math.atan2(*self.point_rotate(p)[::-1])
        #print("abs_shift[1]",self.abs_shift[1],"abs scale",self.abs_scale,"dim[1]",self.dim[1])
        self.abs_shift[1] -= (2*self.abs_shift[1]*self.abs_scale-self.dim[1])/self.abs_scale
        #print("\tas[1]",self.abs_shift[1])
        #print(f"\trotating by {-2*theta*180/np.pi} degrees")
        self.rotate(-2*theta,units='radians')
    def rescale(self):
        self.s=1
    def recenter(self):
        if self.center:
            self.abs_shift = [self.dim[0]/2,self.dim[1]/2]
        else:
            self.abs_shift = [0,0]
    def rotate(self,rot,units=None):
        if units is None:
            units = self.units
        if units[:3].lower()=='deg':
            rot*=np.pi/180
        elif units=='circle':
            rot*=2*np.pi
        elif units=='radians':
            pass
        self.r += rot
    def reset(self):
        self.recenter()
        self.rotate = 0
        self.y_up = self.original_y_up
    def point_shift(self,p):
        return [p[0]+self.abs_shift[0],p[1]+self.abs_shift[1]]
    def point_rotate(self,p):
        r = self.r
        m = np.array([[np.cos(r),-np.sin(r)],[np.sin(r),np.cos(r)]])
        result = np.dot(m,[[p[0]],[p[1]]])
        return [result[0][0],result[1][0]]
    def point_scale(self,p):
        return [self.abs_scale*p[0],self.abs_scale*p[1]]
    def transform(self,p):
        if type(p)==str:
            return self.labels[p]
        base = self.point_scale(self.point_shift(self.point_rotate(p)))
        if self.y_up:
            return [base[0],self.dim[1]-base[1]]
        else:
            return base
    def label(self,p,l):
        self.labels[l]=self.transform(p)
    def background(self,color):
        y,x = self.dim
        #we'll exceed the borders a fair bit for safety
        self.rectangle((-x,-y),(x,y),fill=color,outline=None)
    def line(self,p,q,color='',dashed=False,dashes=5,dash_size = None,dash_frac=0.5,**kwargs):
        if color=='':
            color=self.default_color
        else:
            self.default_color = color
        if dashed:
            P = np.array(p)
            Q = np.array(q)
            qp_dist = np.linalg.norm(Q-P)
            if dash_size is not None:
                assert dashes==5, "Trying to set dashes and dash_vec at once!"
                dash_vec = (Q-P)/qp_dist*dash_size
                dashes = int(qp_dist/dash_size)
            else:
                dash_vec = (Q-P)/dashes
            for i in range(dashes):
                self.line(P+(i+0.5-dash_frac/2)*dash_vec,P+(i+0.5+dash_frac/2)*dash_vec,color,**kwargs)
        else:
            self.draw.line(self.transform(p),self.transform(q),color=self.default_color,**kwargs)
    def path(self,points,color='',**kwargs):
        if color=='':
            color=self.default_color
        else:
            self.default_color = color
        for i in range(len(points)-1):
            self.line(points[i],points[i+1],color,**kwargs)
    def triangle(self,p,q,r,color='',**kwargs):
        self.line(p,q,color,**kwargs)
        self.line(q,r,color,**kwargs)
        self.line(r,p,color,**kwargs)
    def color(self,c):
        self.default_color=c
    def circle(self,p,r,fill='',outline='',**kwargs):
        if outline=='':
            outline=self.default_color
        else:
            self.default_color = outline
        if fill=='':
            fill=self.default_fill
        else:
            self.default_fill = fill
        t = self.transform(p)
        sr = self.abs_scale * r
        self.draw.circle(t,r=sr,fill=fill,outline=outline,**kwargs)
    def poly(self,pts,fill='',outline='',**kwargs):
        if outline=='':
            outline=self.default_color
        else:
            self.default_color = outline
        if fill=='':
            fill=self.default_fill
        else:
            self.default_fill = fill
        self.draw.poly([tuple(self.transform(p)) for p in pts],fill=fill,outline=outline,**kwargs)
    def rectangle(self,corner_1,corner_2,**kwargs):
        a,b=corner_1
        c,d=corner_2
        self.poly([(a,b),(c,b),(c,d),(a,d)],**kwargs)
    def text(self,xy,text,color='',font_size = 20, actual_font = None,anchor='mm',**kwargs):
        #unfortunately we can't rotate well with this
        if color=='':
            color=self.default_color
        if actual_font is None:
            actual_font = ImageFont.truetype("/Users/Drake/Downloads/font.ttf", int(font_size*self.abs_scale))
        self.draw.text(self.transform(xy),text,actual_font=actual_font,color=color,anchor=anchor,**kwargs)
    def text_box(self,xy,text,color='',font_size = 20, actual_font = None,anchor='mm',border=0.,**kwargs):
        if actual_font is None:
            actual_font = ImageFont.truetype("/Users/Drake/Downloads/font.ttf", int(font_size*self.abs_scale))
        a,b,c,d = self.draw.textbbox(xy, text=text, actual_font=actual_font,anchor=anchor)
        #hack to make boxes not be so tall
        a,b,c,d = [x*0.5 for x in [a,b,c,d]]
        border *= font_size
        a,b,c,d = a-border,b-border,c+border,d+border
        self.poly([(a,b),(c,b),(c,d),(a,d)],outline='black',fill='white')
        self.text(xy,text,color,font_size,actual_font,anchor,**kwargs)
    def save(self):
        #to be called for svgs to save the file
        self.draw.save()
    def show(self):
        if self.draw_type == 'imagedraw':
            show(self.img)
        elif self.draw_type == 'svg':
            IPython.display.SVG(self.draw.draw.tostring())
