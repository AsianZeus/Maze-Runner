#Title           : MazeRuner.py
#Description     : Traversing the maze with different algorithms to get the shortest path
#Author          : Akshat Surolia
#Date            : Wdnesday, December 4, 2019
#Usage           : python MazeRunner.py
#Python_version  : 3.7 4 
#=========================================================================================

# Importing Libraries
from tkinter import *
import tkinter.ttk as tpx
from PIL import Image, ImageTk, ImageOps
import threading 
import numpy
import time
from collections import deque


MazePath="SampleMaze.png" # Sample Maze Image

# Algorithm Selection + MazeSolve Button
def RunMaze():
    globals()["algorithm"]=algo.get()
    globals()["Flag"] = globals()["bstart"] & globals()["bend"]
    print("Flag: ",globals()["Flag"])
    main(MazePath)

# Submit Start Node Coordinate
def getstart():
    globals()["startno"]=(slidery.get(), sliderx.get())
    print(f"Start Node: {globals()['startno']}")
    globals()["sposvar"].set(f"Start Position= {globals()['startno']}")
    globals()["bstart"]=True
    ind= globals()["maze"].arpath.index((slidery.get(), sliderx.get()))
    globals()["maze"].start= globals()["maze"].arnode[ind]

# Submit End Node Coordinate    
def getend():
    globals()["endno"]=(slidery.get(), sliderx.get())
    print(f"End Node: {globals()['endno']}")
    globals()["eposvar"].set(f"End Position= {globals()['endno']}")
    globals()["bend"]=True
    ind= globals()["maze"].arpath.index((slidery.get(), sliderx.get()))
    globals()["maze"].end= globals()["maze"].arnode[ind]

# Creating and configuring Frame with Tkinter           
r = Tk() 
r.configure(background='black')
r.geometry("542x660")
r.title('Maze Runner') 

# Adding style and formatting components
style = tpx.Style()
style.configure("TLabel", foreground="white", background="black")
style.configure("TCombobox", foreground="black", background="white")
style.configure("TScale", foreground="white", background="black")

# Configuring combobox for selection of algorithms
algo = tpx.Combobox(r,state="readonly",style="TCombobox")
algo['values']= ("Dijkstra","A Star","Breadth First Search","Depth First Search")
algo.current(0) #set the selected item
algo.grid(column=0, row=0,padx=15, pady=5,sticky=W+E,columnspan=3)

# Configuring Get_Start_Position Button
startbtn= tpx.Button(r,text="Get Start Position",command=getstart)
startbtn.grid(column=1, row=1,padx=5, pady=5,sticky=W)

# Configuring Get_End_Position Button
endbtn= tpx.Button(r,text="Get End Position",command=getend)
endbtn.grid(column=2, row=1,padx=5, pady=5,sticky=W)

# Configuring Solve Button
solvebtn= tpx.Button(r,text="Solve",command=RunMaze,width=30)
solvebtn.grid(column=0, row=2,padx=15, pady=5,sticky=W+E,columnspan=3)

bstart=False
bend=False
startno=(0,0)
endno=(0,0)
Flag=False #Flag = bstart & bend


#Class Maze
class Maze:
    class Node:
        def __init__(self, position):
            self.Position = position
            self.Neighbours = [None, None, None, None] #[below,right,top,left]
        def __repr__(self):
            return f"{self.Position}"

    def __init__(self, im):
        self.arpath=[]
        self.arnode=[]
        width = im.size[0]
        height = im.size[1]
        data = list(im.getdata(0))

        self.start = None
        self.end = None

        topnodes = [None] * width
        count = 0
    # Start row
        if(Flag==False):
            for x in range (1, width - 1):
                if data[x] > 0:
                    self.start = Maze.Node((0,x))
                    topnodes[x] = self.start
                    count += 1
                    self.arpath.append(self.start.Position)
                    self.arnode.append(self.start)
                    break
        j=0
        if(Flag==False): 
            j=1
        for y in range (j, height - 1):

            rowoffset = y * width
            rowaboveoffset = rowoffset - width
            rowbelowoffset = rowoffset + width

            prv = False
            cur = False
            nxt = data[rowoffset + 1] > 0
            leftnode = None

            for x in range (1, width):
                prv = cur
                cur = nxt
                nxt = data[rowoffset + x + 1] > 0

                n = None

                if cur == False:
                    # ON WALL - No action
                    continue
                
                
                if prv == True:
                    if nxt == True:
                        # PATH PATH PATH
                        if data[rowaboveoffset + x] > 0 or data[rowbelowoffset + x] > 0:
                            n = Maze.Node((y,x))
                            leftnode.Neighbours[1] = n
                            n.Neighbours[3] = leftnode
                            leftnode = n
                           
                            if(y==startno[0] and x==startno[1] and Flag):
                                self.start = n
                    else:
                        # PATH PATH WALL
                        n = Maze.Node((y,x))
                        leftnode.Neighbours[1] = n
                        n.Neighbours[3] = leftnode
                        leftnode = None
                                                
                        if(y==startno[0] and x==startno[1] and Flag):
                            self.start = n
                else:
                    if nxt == True:
                        # WALL PATH PATH
                        n = Maze.Node((y,x))
                        leftnode = n

                        if(y==startno[0] and x==startno[1] and Flag):
                            self.start = n
                    else:
                        # WALL PATH WALL
                        if (data[rowaboveoffset + x] == 0) or (data[rowbelowoffset + x] == 0):
                            n = Maze.Node((y,x))
                            
                            if(y==startno[0] and x==startno[1] and Flag):
                                self.start = n
                
                if n != None:
                    if (data[rowaboveoffset + x] > 0):
                        t = topnodes[x]
                        t.Neighbours[2] = n
                        n.Neighbours[0] = t
                    if (data[rowbelowoffset + x] > 0):
                        topnodes[x] = n
                    else:
                        topnodes[x] = None
                    count += 1
                    self.arpath.append(n.Position)
                    self.arnode.append(n)

        # End row
        rowoffset = (height - 1) * width
        for x in range (1, width - 1):
            if data[rowoffset + x] > 0:
                self.end = Maze.Node((height - 1,x))
                t = topnodes[x]
                t.Neighbours[2] = self.end
                self.end.Neighbours[0] = t
                count += 1
                break
            
        if(Flag):
            self.end.Position=(endno)
            
        self.arpath.append(self.end.Position)
        self.arnode.append(self.end)
        self.count = count
        self.width = width
        self.height = height

        

# Opening Image with PIL and creating Maze object
imo=Image.open(MazePath)
maze = Maze(imo)

# Fitting Maze Image to Container Resolution
imglabel = Label(r,height=500,width=500)
imglabel.configure(background='black')
imgtk = ImageTk.PhotoImage(ImageOps.fit(imo, (500,500)))
imglabel.configure(image=imgtk)
imglabel.image = imgtk
imglabel.grid(column=1, row=3,columnspan=4,padx=0)

# Change in position listener of Axis Pointer to denote start and end node 
slider = StringVar()
def callback(*args):
    slider.set(f"Position:: ( {slidery.get()}, {sliderx.get()} )")
    rgb_im = globals()["imo"].convert('RGB')
    impixels = (rgb_im.load())
    red=(255,0,0)
    black=(0,0,0)
    offwhite=(255, 201, 201)
    curpos=(sliderx.get(), slidery.get())
    if(impixels[curpos]!=black):
        if((slidery.get(), sliderx.get()) in maze.arpath): 
            impixels[curpos[0],curpos[1]]=red
        else:
            impixels[curpos[0],curpos[1]]=offwhite
   
    img = rgb_im.resize((500, 500), Image.NEAREST)
    imgtk = ImageTk.PhotoImage(img)
    imglabel.configure(image=imgtk)
    imglabel.image = imgtk

# Configuring x,y variable value change
sliderx = IntVar()
sliderx.set('1')
sliderx.trace("w",callback)
slidery = IntVar()
slidery.trace("w",callback)
slidery.set('1')
slider.set(f"Position: ({sliderx.get()}, {slidery.get()})")

# Configuring labels for Information display
nodec = StringVar()
nodee = StringVar()
pathl = StringVar()
timee = StringVar()
nc = tpx.Label(r,textvariable=nodec,style="TLabel")
ne = tpx.Label(r,textvariable=nodee,style="TLabel")
pl = tpx.Label(r,textvariable=pathl,style="TLabel")
tep= tpx.Label(r,textvariable=timee,style="TLabel")
nc.grid(column=3, row=0,padx=3, pady=3,sticky="w")
ne.grid(column=4, row=0,padx=3, pady=3,sticky="w")
pl.grid(column=3, row=1,padx=3, pady=3,sticky="w")
tep.grid(column=4, row=1,padx=3, pady=3,sticky="w")


# Configuring Y-Axis Slider
yaxis = tpx.Scale(r, from_=0, to_=(imo.size[1]-1),orient=VERTICAL,style="TScale", command=lambda s:slidery.set('%d' % float(s)))
yaxis.set(1)
yaxis.grid(column=0,row=3,sticky=W+N+S,padx=5)

# Image Dimension Label
dimen= tpx.Label(r,text=f"Image Dimnesion: {imo.size[0]} x {imo.size[1]}",style="TLabel")
dimen.grid(column=4, row=2,padx=2, pady=3,sticky=W)

# Position Label
positionlabel = tpx.Label(r, textvariable=slider, style="TLabel") 
positionlabel.grid(column=3, row=2,padx=2, pady=3,sticky=W)

# Configuring X-Axis Slider
xaxis = tpx.Scale(r, from_=0, to_=(imo.size[0]-1),orient=HORIZONTAL,style="TScale", command=lambda s:sliderx.set('%d' % float(s)))
xaxis.grid(column=1,row=4,columnspan=6,sticky=W+E)
yaxis.set(1)

# Setting Start Pointer variable
sposvar = StringVar()
sposvar.set("Start Position= (1,1)")

# Setting End Pointer variable
eposvar = StringVar()
eposvar.set("End Position= (1,1)")

# Creating Start and End position label
spos = tpx.Label(r,textvariable=sposvar,style="TLabel")
epos = tpx.Label(r,textvariable=eposvar,style="TLabel")

spos.grid(column=2,row=5,sticky=W+N+S,padx=5 ,columnspan=2)
epos.grid(column=4,row=5,sticky=W+N+S,padx=5)

# Initializing Values
algorithm="Dijkstra"
bstart=False
bend=False
startno=(0,0)
endno=(0,0)
Flag=False


# Dijkstra Algorithm
def solvedijkstra(maze):
    # Width is used for indexing, total is used for array sizes
    width = maze.width
    total = maze.width * maze.height
    print("\n**Solving with Dijkstra**\n")
    
    # Start node, end node
    start = maze.start
    startpos = start.Position
    end = maze.end
    endpos = end.Position

    visited = [False] * total

    prev = [None] * total

    infinity = float("inf")
    distances = [infinity] * total

    unvisited = FibPQ()
    
    nodeindex = [None] * total

    # Setting the distance to the start to zero and add it into the unvisited queue
    distances[start.Position[0] * width + start.Position[1]] = 0
    startnode = FibHeap.Node(0, start)
    nodeindex[start.Position[0] * width + start.Position[1]] = startnode
    unvisited.insert(startnode)

    # Zero nodes visited, and not completed yet.
    count = 0
    completed = False

    # Begin Dijkstra - continue while there are unvisited nodes in the queue
    while len(unvisited) > 0:
        count += 1

        # Find current shortest path point to explore
        n = unvisited.removeminimum()

        # Current node u, all neighbours will be v
        u = n.value
        upos = u.Position
        uposindex = upos[0] * width + upos[1]

        if distances[uposindex] == infinity:
            break

        # If upos == endpos, we're done!
        if upos == endpos:
            completed = True
            break

        for v in u.Neighbours:
            if v != None:
                vpos = v.Position
                vposindex = vpos[0] * width + vpos[1]

                if visited[vposindex] == False:
                    # The extra distance from where we are (upos) to the neighbour (vpos) - this is manhattan distance
                    d = abs(vpos[0] - upos[0]) + abs(vpos[1] - upos[1])

                    # New path cost to v is distance to u + extra
                    newdistance = distances[uposindex] + d

                    # If this new distance is the new shortest path to v
                    if newdistance < distances[vposindex]:
                        vnode = nodeindex[vposindex]
                        # v isn't already in the priority queue - add it
                        if vnode == None:
                            vnode = FibHeap.Node(newdistance, v)
                            unvisited.insert(vnode)
                            nodeindex[vposindex] = vnode
                            distances[vposindex] = newdistance
                            prev[vposindex] = u
                        # v is already in the queue - decrease its key to re-prioritise it
                        else:
                            unvisited.decreasekey(vnode, newdistance)
                            distances[vposindex] = newdistance
                            prev[vposindex] = u

        visited[uposindex] = True


    # Reconstruct the path. We start at end, and then go prev[end] and follow all the prev[] links until we're back at the start
    
    path = deque()
    current = end
    while (current != None):
        path.appendleft(current)
        current = prev[current.Position[0] * width + current.Position[1]]
    return [path, [count, len(path), completed]]





# A* Algorithm
def solveastar(maze):
    print("\n**Solving with A Star**\n")
    width = maze.width
    total = maze.width * maze.height

    start = maze.start
    startpos = start.Position
    end = maze.end
    endpos = end.Position

    visited = [False] * total
    prev = [None] * total

    infinity = float("inf")
    distances = [infinity] * total

    unvisited = FibPQ()
    
    nodeindex = [None] * total

    distances[start.Position[0] * width + start.Position[1]] = 0
    startnode = FibHeap.Node(0, start)
    nodeindex[start.Position[0] * width + start.Position[1]] = startnode
    unvisited.insert(startnode)

    count = 0
    completed = False

    while len(unvisited) > 0:
        count += 1

        n = unvisited.removeminimum()

        u = n.value
        upos = u.Position
        uposindex = upos[0] * width + upos[1]

        if distances[uposindex] == infinity:
            break

        if upos == endpos:
            completed = True
            break

        for v in u.Neighbours:
            if v != None:
                vpos = v.Position
                vposindex = vpos[0] * width + vpos[1]

                if visited[vposindex] == False:
                    d = abs(vpos[0] - upos[0]) + abs(vpos[1] - upos[1])

                    # New path cost to v is distance to u + extra(g cost).
                    # New distance is the distance of the path from the start, through U, to V.
                    newdistance = distances[uposindex] + d

                    
                    # V to the end. Using manhattan again because A* works well when the g cost and f cost are balanced.                    
                    remaining = abs(vpos[0] - endpos[0]) + abs(vpos[1] - endpos[1])  # Heuristic

                    # Don't include f cost in this first check. We want to know that the path *to* our node V is shortest
                    if newdistance < distances[vposindex]:
                        vnode = nodeindex[vposindex]

                        if vnode == None:
                            # V goes into the priority queue with a cost of g + f. So if it's moving closer to the end, it'll get higher
                            # priority than some other nodes.
                            vnode = FibHeap.Node(newdistance + remaining, v)
                            unvisited.insert(vnode)
                            nodeindex[vposindex] = vnode
                            # The distance *to* the node remains just g, no f included.
                            distances[vposindex] = newdistance
                            prev[vposindex] = u
                        else:
                            # We decrease the node since we've found a new path. But we include the f cost and the distance remaining.
                            unvisited.decreasekey(vnode, newdistance + remaining)
                            # The distance *to* the node remains just g, no f included.
                            distances[vposindex] = newdistance
                            prev[vposindex] = u


        visited[uposindex] = True

    path = deque()
    current = end
    while (current != None):
        path.appendleft(current)
        current = prev[current.Position[0] * width + current.Position[1]]
    return [path, [count, len(path), completed]]






# BFS Algorithm
def solvebfs(maze):
    print("\n**Solving with Breadth First Search**\n")
    start = maze.start
    end = maze.end

    width = maze.width

    queue = deque([start])
    shape = (maze.height, maze.width)
    prev = [None] * (maze.width * maze.height)
    visited = [False] * (maze.width * maze.height)

    count = 0

    completed = False

    visited[start.Position[0] * width + start.Position[1]] = True

    while queue:
        count += 1
        current = queue.pop()

        if current == end:
            completed = True
            break

        for n in current.Neighbours:
            if n != None:
                npos = n.Position[0] * width + n.Position[1]
                if visited[npos] == False:
                    queue.appendleft(n)
                    visited[npos] = True
                    prev[npos] = current

    path = deque()
    current = end
    while (current != None):
        path.appendleft(current)
        current = prev[current.Position[0] * width + current.Position[1]]

    return [path, [count, len(path), completed]]




# DFS Algorithm
def solvedfs(maze):
    start = maze.start
    end = maze.end
    
    width = maze.width
    
    stack = deque([start])
    shape = (maze.height, maze.width)
    prev = [None] * (maze.width * maze.height)
    visited = [False] * (maze.width * maze.height)
    
    count = 0
    
    completed = False
   
    while stack:
        count += 1
        current = stack.pop()
        if current == end:
            completed = True
            break

        visited[current.Position[0] * width + current.Position[1]] = True

        for n in current.Neighbours:
            if n != None:
                npos = n.Position[0] * width + n.Position[1]
                if visited[npos] == False:
                    stack.append(n)
                    visited[npos] = True
                    prev[npos] = current
    path = deque()
    current = end
    while (current != None):
        path.appendleft(current)
        current = prev[current.Position[0] * width + current.Position[1]]
    return [path, [count, len(path), completed]]
    



#Solving Maze with given Algorithm
def solve(input_file):
    maze=globals()["maze"]
    nodec.set(f"Node Count: {maze.count}")
    ts = time.time()
    if(algorithm=="Dijkstra"):
        [result, stats] = solvedijkstra(maze)
    elif (algorithm=="A Star"):     
        [result, stats] = solveastar(maze)
    elif (algorithm=="Breadth First Search"):     
        [result, stats] = solvebfs(maze)
    elif (algorithm=="Depth First Search"):     
        [result, stats] = solvedfs(maze)
    te = time.time()

    total = round(te-ts,6)

    # Print solve stats
    nodee.set(f"Node Explored: {stats[0]}")
    print ("\nNodes explored: ", stats[0])
    if (stats[2]):
        pathl.set(f"Path length: {stats[1]}")
        print ("Path found, length", stats[1])
    else:
        pathl.set(f"No Path Found!")
    timee.set(f"Time Elapsed: {total}")
    print ("Time elapsed: ", total, "\n")
    pathx= [n.Position for n in result]
    return pathx



# Fibonacci Heap implementation for Priority Queue
class FibHeap:

    class Node:
        def __init__(self, key, value):
            # key value degree mark / prev next child parent Key:Priority(Distance) value:Node
            self.key = key
            self.value = value
            self.degree = 0
            self.mark = False
            self.parent = self.child = None
            self.previous = self.next = self

        def issingle(self):
            return self == self.next

        def insert(self, node):
            if node == None:
                return

            self.next.previous = node.previous
            node.previous.next = self.next
            self.next = node
            node.previous = self


        def remove(self):
            self.previous.next = self.next
            self.next.previous = self.previous
            self.next = self.previous = self

        def addchild(self, node):
            if self.child == None:
                self.child = node
            else:
                self.child.insert(node)
            node.parent = self
            node.mark = False
            self.degree += 1

        def removechild(self, node):
            if node.parent != self:
                raise AssertionError("Cannot remove child from a node that is not its parent")

            if node.issingle():
                if self.child != node:
                    raise AssertionError("Cannot remove a node that is not a child")
                self.child = None
            else:
                if self.child == node:
                    self.child = node.next
                node.remove()

            node.parent = None
            node.mark = False
            self.degree -= 1

    def __init__ (self):
        self.minnode = None
        self.count = 0
        self.maxdegree = 0

    def isempty(self):
        return self.count == 0

    def insert(self, node):
        self.count += 1
        self._insertnode(node)

    def _insertnode(self, node):
        if self.minnode == None:
            self.minnode = node
        else:
            self.minnode.insert(node)
            if node.key < self.minnode.key:
                self.minnode = node

    def minimum(self):
        if self.minnode == None:
            raise AssertionError("Cannot return minimum of empty heap")
        return self.minnode

    def merge(self, heap):
        self.minnode.insert(heap.minnode)
        if self.minnode == None or (heap.minnode != None and heap.minnode.key < self.minnode.key):
            self.minnode = heap.minnode
        self.count += heap.count

    def removeminimum(self):
        if self.minnode == None:
            raise AssertionError("Cannot remove from an empty heap")

        removed_node = self.minnode
        self.count -= 1

        # 1: Assign all old root children as new roots
        if self.minnode.child != None:
            c = self.minnode.child

            while True:
                c.parent = None
                c = c.next
                if c == self.minnode.child:
                    break

            self.minnode.child = None
            self.minnode.insert(c)

        # 2.1: If we have removed the last key
        if self.minnode.next == self.minnode:
            if self.count != 0:
                raise AssertionError("Heap error: Expected 0 keys, count is " + str(self.count))
            self.minnode = None
            return removed_node

        # 2.2: Merge any roots with the same degree
        logsize = 100
        degreeroots = [None] * logsize
        self.maxdegree = 0
        currentpointer = self.minnode.next

        while True:
            currentdegree = currentpointer.degree
            current = currentpointer
            currentpointer = currentpointer.next
            while degreeroots[currentdegree] != None:
                other = degreeroots[currentdegree]
                # Swap if required
                if current.key > other.key:
                    temp = other
                    other = current
                    current = temp

                other.remove()
                current.addchild(other)
                degreeroots[currentdegree] = None
                currentdegree += 1

            degreeroots[currentdegree] = current
            if currentpointer == self.minnode:
                break

        # 3: Remove current root and find new minnode
        self.minnode = None
        newmaxdegree = 0
        for d in range (0,logsize):
            if degreeroots[d] != None:
                degreeroots[d].next = degreeroots[d].previous = degreeroots[d]
                self._insertnode(degreeroots[d])
                if (d > newmaxdegree):
                    newmaxdegree = d

        maxdegree = newmaxdegree

        return removed_node


    def decreasekey(self, node, newkey):
        if newkey > node.key:
    
            raise AssertionError("Cannot decrease a key to a greater value")
        elif newkey == node.key:
            return

        node.key = newkey

        parent = node.parent

        if parent == None:
            if newkey < self.minnode.key:
                self.minnode = node
            return
        elif parent.key <= newkey:
            return

        while True:
            parent.removechild(node)
            self._insertnode(node)

            if parent.parent == None:
                break
            elif parent.mark == False:
                parent.mark
                break
            else:
                node = parent
                parent = parent.parent
                continue
            
class FibPQ():
    def __init__(self):
        self.heap = FibHeap()

    def __len__(self):
        return self.heap.count

    def insert(self, node):
        self.heap.insert(node)

    def minimum(self):
        return self.heap.minimum()

    def removeminimum(self):
        return self.heap.removeminimum()

    def decreasekey(self, node, new_priority):
        self.heap.decreasekey(node, new_priority)



# Path Traversal with Gradient Color
def modify(img,result):
    arr = numpy.array(img)
    resultpath = result
    length = len(resultpath)
    for i in range(0, length - 1):
        a = resultpath[i]
        b = resultpath[i+1]
        #print(f"a: {a},b: {b}")
        # Blue - red
        r = int((i / length) * 255)
        px = (r, 0, 255 - r)
        if a[0] == b[0]:
            # Ys equal - horizontal line
            for x in range(min(a[1],b[1]), max(a[1],b[1])):
                arr[a[0],x] = px
        elif a[1] == b[1]:
            # Xs equal - vertical line
            for y in range(min(a[0],b[0]), max(a[0],b[0]) + 1):
                arr[y,a[1]] = px
                
        time.sleep(0.1)
        img = Image.fromarray(arr)
        img = img.resize((500, 500), Image.NEAREST)
        imgtk = ImageTk.PhotoImage(img)
        imglabel.configure(image=imgtk)
        imglabel.image = imgtk




def main(pathim):
    imgx=Image.open(pathim)
    imgn = ImageOps.fit(imgx, (500,500), Image.ANTIALIAS)
    imgtk = ImageTk.PhotoImage(imgn)
    imglabel.configure(image=imgtk)
    imglabel.image = imgtk
    imglabel.grid(column=1, row=3,columnspan=4,padx=0)
    imgx = imgx.convert('RGB')
    result = solve(pathim)
    t1 = threading.Thread(target=modify, args=(imgx,result,)) 
    t1.start()
    
r.mainloop()
