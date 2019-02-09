
def ColorList(colors):
    """
    Function to return specified color by index.
    -------------------------------------------
    Input:
    colors -- list of color indecies( 1,2,3...)
    
    Output:
    cols   -- list with color code
    """
    color_list = ["#9d6d00", "#903ee0", "#11dc79", "#f568ff", "#419500", "#013fb0",
                  "#f2b64c", "#007ae4", "#ff905a", "#33d3e3", "#9e003a", "#019085",
                  "#950065", "#afc98f", "#ff9bfa", "#83221d", "#01668a", "#ff7c7c",
                  "#643561", "#75608a"]

    cols = []
    for i in range(len(colors)):
        indx = colors[i]
        cols.append(color_list[indx])

    return cols
    
