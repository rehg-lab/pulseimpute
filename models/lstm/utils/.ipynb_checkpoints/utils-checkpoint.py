import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition, TransformedBbox, BboxPatch, BboxConnector 
import matplotlib
matplotlib.use('Agg')
cmap = matplotlib.cm.get_cmap('nipy_spectral')

def weights_init(m):
    if hasattr(m, "bias"):
        m.bias.data.fill_(0)
    if hasattr(m, "weight"):
        m.weight.data.fill_(0)

def make_attn_plot(epoch_check_path, epoch, output, X_test, attn_weights):
    
    # valley T wave found in lead 4
    _make_attn_plot_helper(76, 3, "Valley T in Lead 4",epoch_check_path, epoch, output, X_test, attn_weights)

    # peak T wave found in lead 9
    _make_attn_plot_helper(156, 8, "Peak T in Lead 9",epoch_check_path, epoch, output, X_test, attn_weights)

    # random inbetween in all leads
    # pass
    
    # valley S wave found in lead 5
    _make_attn_plot_helper(280, 4, "Valley S in Lead 5",epoch_check_path, epoch, output, X_test, attn_weights)

    # S wave increase in lead 9
    _make_attn_plot_helper(375, 8, "Increase S in Lead 9",epoch_check_path, epoch, output, X_test, attn_weights)

    # Q wave decrease in lead 3
    _make_attn_plot_helper(442, 2, "Decrease Q in Lead 3",epoch_check_path, epoch, output, X_test, attn_weights)

    # valley Q wave found in lead 1
    _make_attn_plot_helper(532, 0, "Valley Q in Lead 1",epoch_check_path, epoch, output, X_test, attn_weights)

    # ST segment in lead 8
    _make_attn_plot_helper(627, 7, "Increase ST in Lead 8",epoch_check_path, epoch, output, X_test, attn_weights)

    # T wave increase in lead 6
    _make_attn_plot_helper(642, 5, "Increase T in Lead 6",epoch_check_path, epoch, output, X_test, attn_weights)

    # T wave decrease in lead 7
    _make_attn_plot_helper(642, 6, "Decrease T in Lead 7",epoch_check_path, epoch, output, X_test, attn_weights)

    # R wave decrease in lead 1
    _make_attn_plot_helper(687, 0, "Decrease R in Lead 1",epoch_check_path, epoch, output, X_test, attn_weights)

    # R wave increase in lead 10
    _make_attn_plot_helper(754, 9, "Increase R in Lead 10",epoch_check_path, epoch, output, X_test, attn_weights)

    # peak R wave found in lead 9
    _make_attn_plot_helper(850, 8, "Peak R in Lead 9",epoch_check_path, epoch, output, X_test, attn_weights)

    # peak P wave found in lead 1
    _make_attn_plot_helper(917, 0, "Peak P in Lead 1",epoch_check_path, epoch, output, X_test, attn_weights)
    
def _make_attn_plot_helper(point, channel, title, epoch_check_path, epoch, output, X_test, attn_weights):
    attn_weights_specific = attn_weights[channel,point,:]
    output_specific = output[-1,:,channel]
    X_test_specific = X_test[-1,:,channel]


    fig = plt.figure(figsize = (35,5))

    x = np.arange(0,1000)

    ax1 = fig.add_subplot(311)
    ax1.plot(x, output_specific, c="b", label = "impute")
    ax1.axvline(x=point, c="red", linewidth=2, zorder=0, clip_on=False)
    plt.legend()

    ax2 = fig.add_subplot(312)
    ax2.plot(x, X_test_specific, c="g", label = "original")
    ax2.axvline(x=point, c="red", linewidth=2, zorder=0, clip_on=False)
    plt.legend()

    ax3 = fig.add_subplot(313)
    ax3.plot(x, attn_weights_specific, c="orange", label = "attention")
    ax3.axvline(x=point, c="red", linewidth=2, zorder=0, clip_on=False)
    plt.legend()

    fig.subplots_adjust(hspace=-.01,top=0.9)
    fig.suptitle(title)
    
    plt.savefig(os.path.join(epoch_check_path, f"attn_{title}_{epoch}.png"),bbox_inches='tight')
    plt.close()
    
def make_impute_plot(epoch_check_path, epoch, output, X_test):
    plt.figure()
    for i in range(12):
        plt.plot(output[-1,:,i], label = i)
    plt.title("Reconstruction")
    plt.legend()
    plt.savefig(os.path.join(epoch_check_path, "reconstruct_epoch_" +  str(epoch) + ".png"))
    plt.close()

    plt.figure()
    for i in range(12):
        plt.plot(X_test[-1,:,i], label = i)
    plt.title("Target")
    plt.legend()
    plt.savefig(os.path.join(epoch_check_path, "target_reconstruct_epoch_" +  str(epoch) + ".png"))
    plt.close()
    
    
    
    ######################################################################################
    # This is on the Target Visualization Code #
    ######################################################################################
   
    plt.figure(figsize = (25,5))
    for i in range(12):
        plt.plot(X_test[-1, 0:1000,i], label = i, zorder=0)
#         plt.plot(X_test[-1, 0:1000,i], label = i, zorder=0,alpha=0.2, c=cmap(i/12))
    plt.legend(loc = 5)

    parent_axes = plt.gca()

    # valley T wave found in lead 4
    ip = InsetPosition(parent_axes,[.01, 1.1, .1 , .3]) # left edge, bottom edge, width, and height in units of the normalized coordinate of the parent axes.
    ax2 = inset_axes(parent_axes, 1, 1)
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(5)
        ax2.spines[axis].set_color('r')
    #add a new axes to the plot and plot whatever you like
    ax2_1 = plt.gcf().add_axes([0,0,1,1])
    ax2_1.plot(output[-1,75:80,3], "k--", label = "I"); 
    ax2_1.plot(X_test[-1,75:80,3], "k-", label = "T"); 
    ax2_1.set_xticks([])
    ax2_1.grid(True)
    ax2_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax2_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    # set the new axes (ax3) to the position of the linked axes
    ax2_1.set_axes_locator(ip)
    # [ending x point, starting x point],[lower height, upper height]
    ax2.plot([80,75], ax2_1.get_ylim())
    # hide the ticks of the linked axes
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_axes_locator(ip)
    # I want to be able to control where the mark is connected to, independently of the data in the ax2.plot call
    mark_inset(parent_axes, ax2, loc1a=4, loc1b=1, loc2a=3, loc2b=2, lw=2)
    ax2.title.set_text("Valley T in Lead 4")


    # peak T wave found in lead 9
    ip = InsetPosition(parent_axes,[.04, -.5, .1 , .3]) # left edge, bottom edge, width, and height in units of the normalized coordinate of the parent axes.
    ax3 = inset_axes(parent_axes, 1, 1)
    ax3.title.set_text("Peak T in Lead 9")
    for axis in ['top','bottom','left','right']:
        ax3.spines[axis].set_linewidth(5)
        ax3.spines[axis].set_color('r')
    ax3_1 = plt.gcf().add_axes([0,0,1,2])
    ax3_1.plot(output[-1,155:160,8], "k--", label = "I"); 
    ax3_1.plot(X_test[-1,155:160,8], "k-", label = "T")
    ax3_1.set_xticks([])
    ax3_1.grid(True)
    ax3_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax3_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    ax3_1.set_axes_locator(ip)
    mark_inset(parent_axes, ax3, loc1a=1, loc1b=4, loc2a=2, loc2b=3, lw=2)
    ax3.plot([160,155],ax3_1.get_ylim())
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_axes_locator(ip)

    # random inbetween in all leads
    ip = InsetPosition(parent_axes,[.15, 1.1, .1 , .3]) # left edge, bottom edge, width, and height in units of the normalized coordinate of the parent axes.
    ax4 = inset_axes(parent_axes, 1, 1)
    ax4.title.set_text("Between Peaks in All Leads")
    for axis in ['top','bottom','left','right']:
        ax4.spines[axis].set_linewidth(5)
        ax4.spines[axis].set_color('k')
    ax4_1 = plt.gcf().add_axes([0,0,1,3])
    ax4_1.plot(output[-1,179:184,:], "k--", label = "I"); 
    ax4_1.plot(X_test[-1,179:184,:], "k-", label = "T")
    ax4_1.set_xticks([])
    ax4_1.grid(True)
    ax4_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax4_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    ax4_1.set_axes_locator(ip)
    mark_inset(parent_axes, ax4, loc1a=4, loc1b=1, loc2a=3, loc2b=2, lw=2)
    ax4.plot([184,179],ax4_1.get_ylim())
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_axes_locator(ip)


    # valley S wave found in lead 5
    ip = InsetPosition(parent_axes,[.18, -.5, .1 , .3]) # left edge, bottom edge, width, and height in units of the normalized coordinate of the parent axes.
    ax5 = inset_axes(parent_axes, 1, 1)
    ax5.title.set_text("Valley S in Lead 5")
    for axis in ['top','bottom','left','right']:
        ax5.spines[axis].set_linewidth(5)
        ax5.spines[axis].set_color('r')
    ax5_1 = plt.gcf().add_axes([0,0,1,4])
    ax5_1.plot(output[-1,278:283,4], "k--", label = "I"); 
    ax5_1.plot(X_test[-1,278:283,4], "k-", label = "T")
    ax5_1.set_xticks([])
    ax5_1.grid(True)
    ax5_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax5_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    ax5_1.set_axes_locator(ip)
    mark_inset(parent_axes, ax5, loc1a=1, loc1b=4, loc2a=2, loc2b=3, lw=2)
    ax5.plot([283,278],ax5_1.get_ylim())
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax5.set_axes_locator(ip)

    # S wave increase in lead 9
    ip = InsetPosition(parent_axes,[.3, 1.1, .1 , .3]) # left edge, bottom edge, width, and height in units of the normalized coordinate of the parent axes.
    ax6 = inset_axes(parent_axes, 1, 1)
    ax6.title.set_text("Increase S in Lead 9")
    for axis in ['top','bottom','left','right']:
        ax6.spines[axis].set_linewidth(5)
        ax6.spines[axis].set_color('k')
    ax6_1 = plt.gcf().add_axes([0,0,1,5])
    ax6_1.plot(output[-1,373:378,8], "k--", label = "I"); 
    ax6_1.plot(X_test[-1,373:378,8], "k-", label = "T")
    ax6_1.set_xticks([])
    ax6_1.grid(True)
    ax6_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax6_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    ax6_1.set_axes_locator(ip)
    mark_inset(parent_axes, ax6, loc1a=4, loc1b=1, loc2a=3, loc2b=2, lw=2)
    ax6.plot([378,373],ax6_1.get_ylim())
    ax6.set_xticks([])
    ax6.set_yticks([])
    ax6.set_axes_locator(ip)

    # Q wave decrease in lead 3
    ip = InsetPosition(parent_axes,[.32, -.5, .1 , .3]) # left edge, bottom edge, width, and height in units of the normalized coordinate of the parent axes.
    ax7 = inset_axes(parent_axes, 1, 1)
    ax7.title.set_text("Decrease Q in Lead 3")
    for axis in ['top','bottom','left','right']:
        ax7.spines[axis].set_linewidth(5)
        ax7.spines[axis].set_color('k')
    ax7_1 = plt.gcf().add_axes([0,0,1,6])
    ax7_1.plot(output[-1,440:445,2], "k--", label = "I"); 
    ax7_1.plot(X_test[-1,440:445,2], "k-", label = "T")
    ax7_1.set_xticks([])
    ax7_1.grid(True)
    ax7_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax7_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    ax7_1.set_axes_locator(ip)
    mark_inset(parent_axes, ax7, loc1a=1, loc1b=4, loc2a=2, loc2b=3, lw=2)
    ax7.plot([440,445],ax7_1.get_ylim())
    ax7.set_xticks([])
    ax7.set_yticks([])
    ax7.set_axes_locator(ip)

    # valley Q wave found in lead 1
    ip = InsetPosition(parent_axes,[.45, 1.1, .1 , .3]) # left edge, bottom edge, width, and height in units of the normalized coordinate of the parent axes.
    ax8 = inset_axes(parent_axes, 1, 1)
    ax8.title.set_text("Valley Q in Lead 1")
    for axis in ['top','bottom','left','right']:
        ax8.spines[axis].set_linewidth(5)
        ax8.spines[axis].set_color('r')
    ax8_1 = plt.gcf().add_axes([0,0,1,7])
    ax8_1.plot(output[-1,530:535,0], "k--", label = "I"); 
    ax8_1.plot(X_test[-1,530:535,0], "k-", label = "T")
    ax8_1.set_xticks([])
    ax8_1.grid(True)
    ax8_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax8_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    ax8_1.set_axes_locator(ip)
    mark_inset(parent_axes, ax8, loc1a=4, loc1b=1, loc2a=3, loc2b=2, lw=2)
    ax8.plot([530,535],ax8_1.get_ylim())
    ax8.set_xticks([])
    ax8.set_yticks([])
    ax8.set_axes_locator(ip)


    # ST segment in lead 8
    ip = InsetPosition(parent_axes,[.45, -.5, .1 , .3]) 
    ax9 = inset_axes(parent_axes, 1, 1)
    ax9.title.set_text("Increase ST in Lead 8")
    for axis in ['top','bottom','left','right']:
        ax9.spines[axis].set_linewidth(5)
        ax9.spines[axis].set_color('k')
    ax9_1 = plt.gcf().add_axes([0,0,1,8])
    ax9_1.plot(output[-1,625:630,7], "k--", label = "I"); 
    ax9_1.plot(X_test[-1,625:630,7], "k-", label = "T")
    ax9_1.set_xticks([])
    ax9_1.grid(True)
    ax9_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax8_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    ax9_1.set_axes_locator(ip)
    mark_inset(parent_axes, ax9, loc1a=1, loc1b=4, loc2a=2, loc2b=3, lw=2)
    ax9.plot([630,625],ax9_1.get_ylim())
    ax9.set_xticks([])
    ax9.set_yticks([])
    ax9.set_axes_locator(ip)

    # T wave increase in lead 6
    ip = InsetPosition(parent_axes,[.58, 1.1, .1 , .3]) # left edge, bottom edge, width, and height in units of the normalized coordinate of the parent axes.
    ax10 = inset_axes(parent_axes, 1, 1)
    ax10.title.set_text("Increase T in Lead 6")
    for axis in ['top','bottom','left','right']:
        ax10.spines[axis].set_linewidth(5)
        ax10.spines[axis].set_color('k')
    ax10_1 = plt.gcf().add_axes([0,0,1,9])
    ax10_1.plot(output[-1,640:645,5], "k--", label = "I"); 
    ax10_1.plot(X_test[-1,640:645,5], "k-", label = "T")
    ax10_1.set_xticks([])
    ax10_1.grid(True)
    ax10_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax10_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    ax10_1.set_axes_locator(ip)
    mark_inset(parent_axes, ax10, loc1a=4, loc1b=1, loc2a=3, loc2b=2, lw=2)
    ax10.plot([645,640],ax10_1.get_ylim())
    ax10.set_xticks([])
    ax10.set_yticks([])
    ax10.set_axes_locator(ip)


    # T wave decrease in lead 7
    ip = InsetPosition(parent_axes,[.58, -.5, .1 , .3]) # left edge, bottom edge, width, and height in units of the normalized coordinate of the parent axes.
    ax11 = inset_axes(parent_axes, 1, 1)
    ax11.title.set_text("Decrease T in Lead 7")
    for axis in ['top','bottom','left','right']:
        ax11.spines[axis].set_linewidth(5)
        ax11.spines[axis].set_color('k')
    ax11_1 = plt.gcf().add_axes([0,0,1,10])
    ax11_1.plot(output[-1,640:645,6], "k--", label = "I"); 
    ax11_1.plot(X_test[-1,640:645,6], "k-", label = "T")
    ax11_1.set_xticks([])
    ax11_1.grid(True)
    ax11_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax11_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    ax11_1.set_axes_locator(ip)
    mark_inset(parent_axes, ax11, loc1a=1, loc1b=4, loc2a=2, loc2b=3, lw=2)
    ax11.plot([645,640],ax11_1.get_ylim())
    ax11.set_xticks([])
    ax11.set_yticks([])
    ax11.set_axes_locator(ip)


    # R wave decrease in lead 1
    ip = InsetPosition(parent_axes,[.7, 1.1, .1 , .3]) # left edge, bottom edge, width, and height in units of the normalized coordinate of the parent axes.
    ax12 = inset_axes(parent_axes, 1, 1)
    ax12.title.set_text("Decrease R in Lead 1")
    for axis in ['top','bottom','left','right']:
        ax12.spines[axis].set_linewidth(5)
        ax12.spines[axis].set_color('k')
    ax12_1 = plt.gcf().add_axes([0,0,1,11])
    ax12_1.plot(output[-1,685:690,0], "k--", label = "I"); 
    ax12_1.plot(X_test[-1,685:690,0], "k-", label = "T")
    ax12_1.set_xticks([])
    ax12_1.grid(True)
    ax12_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax12_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    ax12_1.set_axes_locator(ip)
    mark_inset(parent_axes, ax12, loc1a=4, loc1b=1, loc2a=3, loc2b=2, lw=2)
    ax12.plot([690,685],ax12_1.get_ylim())
    ax12.set_xticks([])
    ax12.set_yticks([])
    ax12.set_axes_locator(ip)

    # R wave increase in lead 10
    ip = InsetPosition(parent_axes,[.72, -.5, .1 , .3]) # left edge, bottom edge, width, and height in units of the normalized coordinate of the parent axes.
    ax13 = inset_axes(parent_axes, 1, 1)
    ax13.title.set_text("Increase R in Lead 10")
    for axis in ['top','bottom','left','right']:
        ax13.spines[axis].set_linewidth(5)
        ax13.spines[axis].set_color('k')
    ax13_1 = plt.gcf().add_axes([0,0,1,12])
    ax13_1.plot(output[-1,752:757,9], "k--", label = "I"); 
    ax13_1.plot(X_test[-1,752:757,9], "k-", label = "T")
    ax13_1.set_xticks([])
    ax13_1.grid(True)
    ax13_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax13_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    ax13_1.set_axes_locator(ip)
    mark_inset(parent_axes, ax13, loc1a=1, loc1b=4, loc2a=2, loc2b=3, lw=2)
    ax13.plot([757,752],ax13_1.get_ylim())
    ax13.set_xticks([])
    ax13.set_yticks([])
    ax13.set_axes_locator(ip)

    # peak R wave found in lead 9
    ip = InsetPosition(parent_axes,[.85, 1.1, .1 , .3]) # left edge, bottom edge, width, and height in units of the normalized coordinate of the parent axes.
    ax14 = inset_axes(parent_axes, 1, 1)
    ax14.title.set_text("Peak R in Lead 9")
    for axis in ['top','bottom','left','right']:
        ax14.spines[axis].set_linewidth(5)
        ax14.spines[axis].set_color('r')
    ax14_1 = plt.gcf().add_axes([0,0,1,13])
    ax14_1.plot(output[-1,848:853,8], "k--", label = "I"); 
    ax14_1.plot(X_test[-1,848:853,8], "k-", label = "T")
    ax14_1.set_xticks([])
    ax14_1.grid(True)
    ax14_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax14_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    ax14_1.set_axes_locator(ip)
    mark_inset(parent_axes, ax14, loc1a=4, loc1b=1, loc2a=3, loc2b=2, lw=2)
    ax14.plot([853,848],ax14_1.get_ylim())
    ax14.set_xticks([])
    ax14.set_yticks([])
    ax14.set_axes_locator(ip)


    # peak P wave found in lead 1
    ip = InsetPosition(parent_axes,[.87, -.5, .1 , .3]) # left edge, bottom edge, width, and height in units of the normalized coordinate of the parent axes.
    ax15 = inset_axes(parent_axes, 1, 1)
    ax15.title.set_text("Peak P in Lead 1")
    for axis in ['top','bottom','left','right']:
        ax15.spines[axis].set_linewidth(5)
        ax15.spines[axis].set_color('r')
    ax15_1 = plt.gcf().add_axes([0,0,1,14])
    ax15_1.plot(output[-1,915:920,0], "k--", label = "I"); 
    ax15_1.plot(X_test[-1,915:920,0], "k-", label = "T")
    ax15_1.set_xticks([])
    ax15_1.grid(True)
    ax15_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax15_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    ax15_1.set_axes_locator(ip)
    mark_inset(parent_axes, ax15, loc1a=1, loc1b=4, loc2a=2, loc2b=3, lw=2)
    ax15.plot([920,915],ax15_1.get_ylim())
    ax15.set_xticks([])
    ax15.set_yticks([])
    ax15.set_axes_locator(ip)

    plt.savefig(os.path.join(epoch_check_path, "impute_ontarget_epoch_" +  str(epoch) + ".png"),bbox_inches='tight')
    plt.close()

    
    
    
    
    ######################################################################################
    # This is on the Reconstruction Visuzation Code #
    ######################################################################################
    
    plt.figure(figsize = (25,5))
    for i in range(12):
        plt.plot(output[-1, 0:1000,i], label = i, zorder=0)
#         plt.plot(output[-1, 0:1000,i], label = i, zorder=0,alpha=0.2, c=cmap(i/12))
    plt.legend(loc = 5)

    parent_axes = plt.gca()

    # valley T wave found in lead 4
    ip = InsetPosition(parent_axes,[.01, 1.1, .1 , .3]) # left edge, bottom edge, width, and height in units of the normalized coordinate of the parent axes.
    ax2 = inset_axes(parent_axes, 1, 1)
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(5)
        ax2.spines[axis].set_color('r')
    #add a new axes to the plot and plot whatever you like
    ax2_1 = plt.gcf().add_axes([0,0,1,1])
    ax2_1.plot(output[-1,75:80,3], "k--", label = "I"); 
    ax2_1.plot(X_test[-1,75:80,3], "k-", label = "T"); 
    ax2_1.set_xticks([])
    ax2_1.grid(True)
    ax2_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax2_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    # set the new axes (ax3) to the position of the linked axes
    ax2_1.set_axes_locator(ip)
    # [ending x point, starting x point],[lower height, upper height]
    ax2.plot([80,75], ax2_1.get_ylim())
    # hide the ticks of the linked axes
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_axes_locator(ip)
    # I want to be able to control where the mark is connected to, independently of the data in the ax2.plot call
    mark_inset(parent_axes, ax2, loc1a=4, loc1b=1, loc2a=3, loc2b=2, lw=2)
    ax2.title.set_text("Valley T in Lead 4")


    # peak T wave found in lead 9
    ip = InsetPosition(parent_axes,[.04, -.5, .1 , .3]) # left edge, bottom edge, width, and height in units of the normalized coordinate of the parent axes.
    ax3 = inset_axes(parent_axes, 1, 1)
    ax3.title.set_text("Peak T in Lead 9")
    for axis in ['top','bottom','left','right']:
        ax3.spines[axis].set_linewidth(5)
        ax3.spines[axis].set_color('r')
    ax3_1 = plt.gcf().add_axes([0,0,1,2])
    ax3_1.plot(output[-1,155:160,8], "k--", label = "I"); 
    ax3_1.plot(X_test[-1,155:160,8], "k-", label = "T")
    ax3_1.set_xticks([])
    ax3_1.grid(True)
    ax3_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax3_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    ax3_1.set_axes_locator(ip)
    mark_inset(parent_axes, ax3, loc1a=1, loc1b=4, loc2a=2, loc2b=3, lw=2)
    ax3.plot([160,155],ax3_1.get_ylim())
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_axes_locator(ip)

    # random inbetween in all leads
    ip = InsetPosition(parent_axes,[.15, 1.1, .1 , .3]) # left edge, bottom edge, width, and height in units of the normalized coordinate of the parent axes.
    ax4 = inset_axes(parent_axes, 1, 1)
    ax4.title.set_text("Between Peaks in All Leads")
    for axis in ['top','bottom','left','right']:
        ax4.spines[axis].set_linewidth(5)
        ax4.spines[axis].set_color('k')
    ax4_1 = plt.gcf().add_axes([0,0,1,3])
    ax4_1.plot(output[-1,179:184,:], "k--", label = "I"); 
    ax4_1.plot(X_test[-1,179:184,:], "k-", label = "T")
    ax4_1.set_xticks([])
    ax4_1.grid(True)
    ax4_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax4_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    ax4_1.set_axes_locator(ip)
    mark_inset(parent_axes, ax4, loc1a=4, loc1b=1, loc2a=3, loc2b=2, lw=2)
    ax4.plot([184,179],ax4_1.get_ylim())
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_axes_locator(ip)


    # valley S wave found in lead 5
    ip = InsetPosition(parent_axes,[.18, -.5, .1 , .3]) # left edge, bottom edge, width, and height in units of the normalized coordinate of the parent axes.
    ax5 = inset_axes(parent_axes, 1, 1)
    ax5.title.set_text("Valley S in Lead 5")
    for axis in ['top','bottom','left','right']:
        ax5.spines[axis].set_linewidth(5)
        ax5.spines[axis].set_color('r')
    ax5_1 = plt.gcf().add_axes([0,0,1,4])
    ax5_1.plot(output[-1,278:283,4], "k--", label = "I"); 
    ax5_1.plot(X_test[-1,278:283,4], "k-", label = "T")
    ax5_1.set_xticks([])
    ax5_1.grid(True)
    ax5_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax5_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    ax5_1.set_axes_locator(ip)
    mark_inset(parent_axes, ax5, loc1a=1, loc1b=4, loc2a=2, loc2b=3, lw=2)
    ax5.plot([283,278],ax5_1.get_ylim())
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax5.set_axes_locator(ip)

    # S wave increase in lead 9
    ip = InsetPosition(parent_axes,[.3, 1.1, .1 , .3]) # left edge, bottom edge, width, and height in units of the normalized coordinate of the parent axes.
    ax6 = inset_axes(parent_axes, 1, 1)
    ax6.title.set_text("Increase S in Lead 9")
    for axis in ['top','bottom','left','right']:
        ax6.spines[axis].set_linewidth(5)
        ax6.spines[axis].set_color('k')
    ax6_1 = plt.gcf().add_axes([0,0,1,5])
    ax6_1.plot(output[-1,373:378,8], "k--", label = "I"); 
    ax6_1.plot(X_test[-1,373:378,8], "k-", label = "T")
    ax6_1.set_xticks([])
    ax6_1.grid(True)
    ax6_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax6_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    ax6_1.set_axes_locator(ip)
    mark_inset(parent_axes, ax6, loc1a=4, loc1b=1, loc2a=3, loc2b=2, lw=2)
    ax6.plot([378,373],ax6_1.get_ylim())
    ax6.set_xticks([])
    ax6.set_yticks([])
    ax6.set_axes_locator(ip)

    # Q wave decrease in lead 3
    ip = InsetPosition(parent_axes,[.32, -.5, .1 , .3]) # left edge, bottom edge, width, and height in units of the normalized coordinate of the parent axes.
    ax7 = inset_axes(parent_axes, 1, 1)
    ax7.title.set_text("Decrease Q in Lead 3")
    for axis in ['top','bottom','left','right']:
        ax7.spines[axis].set_linewidth(5)
        ax7.spines[axis].set_color('k')
    ax7_1 = plt.gcf().add_axes([0,0,1,6])
    ax7_1.plot(output[-1,440:445,2], "k--", label = "I"); 
    ax7_1.plot(X_test[-1,440:445,2], "k-", label = "T")
    ax7_1.set_xticks([])
    ax7_1.grid(True)
    ax7_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax7_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    ax7_1.set_axes_locator(ip)
    mark_inset(parent_axes, ax7, loc1a=1, loc1b=4, loc2a=2, loc2b=3, lw=2)
    ax7.plot([440,445],ax7_1.get_ylim())
    ax7.set_xticks([])
    ax7.set_yticks([])
    ax7.set_axes_locator(ip)

    # valley Q wave found in lead 1
    ip = InsetPosition(parent_axes,[.45, 1.1, .1 , .3]) # left edge, bottom edge, width, and height in units of the normalized coordinate of the parent axes.
    ax8 = inset_axes(parent_axes, 1, 1)
    ax8.title.set_text("Valley Q in Lead 1")
    for axis in ['top','bottom','left','right']:
        ax8.spines[axis].set_linewidth(5)
        ax8.spines[axis].set_color('r')
    ax8_1 = plt.gcf().add_axes([0,0,1,7])
    ax8_1.plot(output[-1,530:535,0], "k--", label = "I"); 
    ax8_1.plot(X_test[-1,530:535,0], "k-", label = "T")
    ax8_1.set_xticks([])
    ax8_1.grid(True)
    ax8_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax8_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    ax8_1.set_axes_locator(ip)
    mark_inset(parent_axes, ax8, loc1a=4, loc1b=1, loc2a=3, loc2b=2, lw=2)
    ax8.plot([530,535],ax8_1.get_ylim())
    ax8.set_xticks([])
    ax8.set_yticks([])
    ax8.set_axes_locator(ip)


    # ST segment in lead 8
    ip = InsetPosition(parent_axes,[.45, -.5, .1 , .3]) 
    ax9 = inset_axes(parent_axes, 1, 1)
    ax9.title.set_text("Increase ST in Lead 8")
    for axis in ['top','bottom','left','right']:
        ax9.spines[axis].set_linewidth(5)
        ax9.spines[axis].set_color('k')
    ax9_1 = plt.gcf().add_axes([0,0,1,8])
    ax9_1.plot(output[-1,625:630,7], "k--", label = "I"); 
    ax9_1.plot(X_test[-1,625:630,7], "k-", label = "T")
    ax9_1.set_xticks([])
    ax9_1.grid(True)
    ax9_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax8_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    ax9_1.set_axes_locator(ip)
    mark_inset(parent_axes, ax9, loc1a=1, loc1b=4, loc2a=2, loc2b=3, lw=2)
    ax9.plot([630,625],ax9_1.get_ylim())
    ax9.set_xticks([])
    ax9.set_yticks([])
    ax9.set_axes_locator(ip)

    # T wave increase in lead 6
    ip = InsetPosition(parent_axes,[.58, 1.1, .1 , .3]) # left edge, bottom edge, width, and height in units of the normalized coordinate of the parent axes.
    ax10 = inset_axes(parent_axes, 1, 1)
    ax10.title.set_text("Increase T in Lead 6")
    for axis in ['top','bottom','left','right']:
        ax10.spines[axis].set_linewidth(5)
        ax10.spines[axis].set_color('k')
    ax10_1 = plt.gcf().add_axes([0,0,1,9])
    ax10_1.plot(output[-1,640:645,5], "k--", label = "I"); 
    ax10_1.plot(X_test[-1,640:645,5], "k-", label = "T")
    ax10_1.set_xticks([])
    ax10_1.grid(True)
    ax10_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax10_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    ax10_1.set_axes_locator(ip)
    mark_inset(parent_axes, ax10, loc1a=4, loc1b=1, loc2a=3, loc2b=2, lw=2)
    ax10.plot([645,640],ax10_1.get_ylim())
    ax10.set_xticks([])
    ax10.set_yticks([])
    ax10.set_axes_locator(ip)


    # T wave decrease in lead 7
    ip = InsetPosition(parent_axes,[.58, -.5, .1 , .3]) # left edge, bottom edge, width, and height in units of the normalized coordinate of the parent axes.
    ax11 = inset_axes(parent_axes, 1, 1)
    ax11.title.set_text("Decrease T in Lead 7")
    for axis in ['top','bottom','left','right']:
        ax11.spines[axis].set_linewidth(5)
        ax11.spines[axis].set_color('k')
    ax11_1 = plt.gcf().add_axes([0,0,1,10])
    ax11_1.plot(output[-1,640:645,6], "k--", label = "I"); 
    ax11_1.plot(X_test[-1,640:645,6], "k-", label = "T")
    ax11_1.set_xticks([])
    ax11_1.grid(True)
    ax11_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax11_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    ax11_1.set_axes_locator(ip)
    mark_inset(parent_axes, ax11, loc1a=1, loc1b=4, loc2a=2, loc2b=3, lw=2)
    ax11.plot([645,640],ax11_1.get_ylim())
    ax11.set_xticks([])
    ax11.set_yticks([])
    ax11.set_axes_locator(ip)


    # R wave decrease in lead 1
    ip = InsetPosition(parent_axes,[.7, 1.1, .1 , .3]) # left edge, bottom edge, width, and height in units of the normalized coordinate of the parent axes.
    ax12 = inset_axes(parent_axes, 1, 1)
    ax12.title.set_text("Decrease R in Lead 1")
    for axis in ['top','bottom','left','right']:
        ax12.spines[axis].set_linewidth(5)
        ax12.spines[axis].set_color('k')
    ax12_1 = plt.gcf().add_axes([0,0,1,11])
    ax12_1.plot(output[-1,685:690,0], "k--", label = "I"); 
    ax12_1.plot(X_test[-1,685:690,0], "k-", label = "T")
    ax12_1.set_xticks([])
    ax12_1.grid(True)
    ax12_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax12_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    ax12_1.set_axes_locator(ip)
    mark_inset(parent_axes, ax12, loc1a=4, loc1b=1, loc2a=3, loc2b=2, lw=2)
    ax12.plot([690,685],ax12_1.get_ylim())
    ax12.set_xticks([])
    ax12.set_yticks([])
    ax12.set_axes_locator(ip)

    # R wave increase in lead 10
    ip = InsetPosition(parent_axes,[.72, -.5, .1 , .3]) # left edge, bottom edge, width, and height in units of the normalized coordinate of the parent axes.
    ax13 = inset_axes(parent_axes, 1, 1)
    ax13.title.set_text("Increase R in Lead 10")
    for axis in ['top','bottom','left','right']:
        ax13.spines[axis].set_linewidth(5)
        ax13.spines[axis].set_color('k')
    ax13_1 = plt.gcf().add_axes([0,0,1,12])
    ax13_1.plot(output[-1,752:757,9], "k--", label = "I"); 
    ax13_1.plot(X_test[-1,752:757,9], "k-", label = "T")
    ax13_1.set_xticks([])
    ax13_1.grid(True)
    ax13_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax13_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    ax13_1.set_axes_locator(ip)
    mark_inset(parent_axes, ax13, loc1a=1, loc1b=4, loc2a=2, loc2b=3, lw=2)
    ax13.plot([757,752],ax13_1.get_ylim())
    ax13.set_xticks([])
    ax13.set_yticks([])
    ax13.set_axes_locator(ip)

    # peak R wave found in lead 9
    ip = InsetPosition(parent_axes,[.85, 1.1, .1 , .3]) # left edge, bottom edge, width, and height in units of the normalized coordinate of the parent axes.
    ax14 = inset_axes(parent_axes, 1, 1)
    ax14.title.set_text("Peak R in Lead 9")
    for axis in ['top','bottom','left','right']:
        ax14.spines[axis].set_linewidth(5)
        ax14.spines[axis].set_color('r')
    ax14_1 = plt.gcf().add_axes([0,0,1,13])
    ax14_1.plot(output[-1,848:853,8], "k--", label = "I"); 
    ax14_1.plot(X_test[-1,848:853,8], "k-", label = "T")
    ax14_1.set_xticks([])
    ax14_1.grid(True)
    ax14_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax14_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    ax14_1.set_axes_locator(ip)
    mark_inset(parent_axes, ax14, loc1a=4, loc1b=1, loc2a=3, loc2b=2, lw=2)
    ax14.plot([853,848],ax14_1.get_ylim())
    ax14.set_xticks([])
    ax14.set_yticks([])
    ax14.set_axes_locator(ip)


    # peak P wave found in lead 1
    ip = InsetPosition(parent_axes,[.87, -.5, .1 , .3]) # left edge, bottom edge, width, and height in units of the normalized coordinate of the parent axes.
    ax15 = inset_axes(parent_axes, 1, 1)
    ax15.title.set_text("Peak P in Lead 1")
    for axis in ['top','bottom','left','right']:
        ax15.spines[axis].set_linewidth(5)
        ax15.spines[axis].set_color('r')
    ax15_1 = plt.gcf().add_axes([0,0,1,14])
    ax15_1.plot(output[-1,915:920,0], "k--", label = "I"); 
    ax15_1.plot(X_test[-1,915:920,0], "k-", label = "T")
    ax15_1.set_xticks([])
    ax15_1.grid(True)
    ax15_1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    for tick in ax15_1.yaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    ax15_1.set_axes_locator(ip)
    mark_inset(parent_axes, ax15, loc1a=1, loc1b=4, loc2a=2, loc2b=3, lw=2)
    ax15.plot([920,915],ax15_1.get_ylim())
    ax15.set_xticks([])
    ax15.set_yticks([])
    ax15.set_axes_locator(ip)

    plt.savefig(os.path.join(epoch_check_path, "impute_onreconstruct_epoch_" +  str(epoch) + ".png"),bbox_inches='tight')
    plt.close()

def mark_inset(parent_axes, inset_axes, loc1a=1, loc1b=1, loc2a=2, loc2b=2, **kwargs):
    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)

    pp = BboxPatch(rect, fill=False, **kwargs)
    parent_axes.add_patch(pp)

    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1a, loc2=loc1b, **kwargs)
    inset_axes.add_patch(p1)
    p1.set_clip_on(False)
    p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2a, loc2=loc2b, **kwargs)
    inset_axes.add_patch(p2)
    p2.set_clip_on(False)

    return pp, p1, p2