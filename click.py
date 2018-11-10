from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import ImageTk

#convert between window and canvas coordinates
#event2canvas = lambda e, c: (c.canvasx(e.x), c.canvasy(e.y))

if __name__ == "__main__":
    root = Tk()
    root.minsize(500,500)

    #setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E+W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N+S)
    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=BOTH,expand=1)

    #adding the image
    File = askopenfilename(parent=root, initialdir="M:/",title='Choose an image.')
    print("opening %s" % File)
    img = ImageTk.PhotoImage(file=File)
    canvas.create_image(0,0,image=img,anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))

    f= open("%.txt" % File, "w+")
    
    #functions to be called when mouse is clicked
    def petitemark(event):
       f.write(event.x, event.y, '\n')
       #marks petite colonies green
       canvas.create_oval(event.x-1,event.y+1,event.x+1,event.y-1,outline='green')
    def grandemark(event):
        f.write(event.x, event.y, '\n')
        #marks grande colonies red
        canvas.create_oval(event.x-1,event.y+1,event.x+1,event.y-1,outline='red')
    def close(event):
        root.destroy()
        f.close()
    
    #mouseclick events: left click to mark petite, right click to mark grande, enter to finish
    canvas.bind("<Button-1>",petitemark)
    canvas.bind("<Button-3>",grandemark)
    canvas.bind("<Return>", close)

    root.mainloop()
