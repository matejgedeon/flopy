import warnings

import numpy as np
import os

from ..pakbase import Package
from ..utils import MfList, Transient2d, Util2d, Util3d

class McpPackage:
    def __init__(self, label="", instance=None, needTFstr=False):
        self.label = label
        self.instance = instance
        self.needTFstr = needTFstr
        self.TFstr = " F"
        if self.instance is not None:
            self.TFstr = " T"


class MthMcp(Package):
    """
    MTHP Multi-Component Reactive Transport (using PHREEQC) Package Class.

    Parameters
    ----------
    model : model object
        The model object (of type :class:`flopy.mthp.mth.Mthp`) to which
        this package will be added.
    speciesfile : int
        Flag to indicate if the database and components are defined in a 
        separate file. This is preferable when the same database and 
        components are used in different packages (e.g., the groundwater
        flow and the UZF or the HYDRUS package). 0: information for the 
        species file is included in this file; 1: information for the 
        species file is defined in a separate file
    globalfile : int
        Flag to indicate that a global external geochemical input file for
        all packages will be used. This is used when user adds new components,
        scripts or rate equations to be used in different packages (e.g., 
        the groundwater flow and the UZF or the HYDRUS package). 0: no global
        external geochemical input file will be used. 1: a global external 
        geochemical input file will be used.
    nhpfile : int
        Specifies how many external geochemical input files will be used for
        the package UFZ or HYDRUS.
    ftemp : int
        Flag indicating how temperature is defined. 0: constant spatial-
        temporal temperature field is used (default); 1: constant temporal 
        temperature field is used; 2: one of the components is treated as 
        'temperature' to simulate heat transport.
    basenr : int
        Linking information on concentration to cell numbers in the geochemical
        model. If BASENR < NLAY*NCOL*NROW, then BASENR is changed to be equal 
        to NLAY*NCOL*NROW
    speciesfilename : string
        Used if SPECIESFILE = 1. SPECIESFILENAME is the path and file name for
        the external file with information on the database and the component 
        names. The file consists of the following records:
            1	Record:	(Optional) string starting with "Pcp_File"
                If the first line starts with "Pcp_File", the line will be 
                neglected.
            2	Record: “relative” | DATABASE. If the word “relative” is gvien,
                it will be assumed that a relative path to the DATABASE is 
                given (in item 3). If it is different from “relative”, it is 
                assumed that it is DATABASE (absolute path) and item 3 should 
                not be defined.
            3	Record:	DATABASE. DATABASE is a string for path and filename
                of the geochemical database. 
            4	Record:	COMPONENT (read item 4 for each component). Every
                COMPONENT is a string of the component name as used in the 
                thermodynamic database. In addition to the chemical elements 
                as components (e.g., Na, Ca, S(6), ...), also Total_O, Total_H,
                Charge, and Heat are valid names.
    database : string
        Either a keyword “relative” or DATABASE. If the keyword “relative” is
        gvien, it will be assumed that a relative path to the DATABASE is given
        in next line. If it is different from “relative”, it is assumed that 
        the absolute path to the DATABASE is specified.
    component : list
        Every COMPONENT is a string of the component name as used in the 
        thermodynamic database. In addition to the chemical elements as 
        components (e.g., Na, Ca, S(6), ...), also Total_O, Total_H, Charge,
        and Heat are valid names. 
    globalfilename : string
        Specifies GLOBAL,a string for path and filename of the global 
        geochemical input file used for ground water and unsaturated 
        geochemical calculations (e.g., with HYDRUS). This file typically 
        contains additions to the database common for all considered systems 
        and selected output statements.           
    gwfile : string
        GWFILE is a string for path and filename of the geochemical input file
        used for MT3D-USGS related geochemical input  (for PHREEQC). This file
        contains solution numbers related to initial conditions, sources and 
        sinks, boundary concentrations, and geochemical definitions 
        (equilibrium phases, exchange, surface complexation, solid solutions, 
         kinetics).
    hpfile : string
        Used if nhpfile > 0. HPFILE is a string for path and filename of the
        geochemical input file (for PHREEQC) used for the unsaturated zone 
        conditions when HYDRUS is used. This file contains solution numbers 
        related to initial conditions, sources and sinks, boundary 
        concentrations, and geochemical definitions (equilibrium phases, 
        exchange, surface complexation, solid solutions, kinetics).
    outputpath : string
        OUTPUTPATH is a string for the (relative) path where the output of the 
        geochemical calculations is saved.
    ichemdistr : int or array of integers (nlay, nrow, ncol)
        The initial geochemical condition. The value refers to the numbers in 
        the geochemical input files (typically GWFILE) where the numbers are 
        defined as ICHEMDISTR + BASNR. The geochemical condition consists of a 
        solution definition and, depending on the geochemical model, of 
        equilibrium phases, exchange, surface complexation, solid solutions, 
        and/or kinetics.    
    tempetarure : float
        TEMPERATURE is the temperature in the simulation domain, and of all 
        sources and sinks. 
    extension : string
        Filename extension (default is 'mcp')
    unitnumber : int
        File unit number (default is None).
    filenames : str or list of str
        Filenames to use for the package. If filenames=None the package name
        will be created using the model name and package extension. If a
        single string is passed the package will be set to the string.
        Default is None.
    createspcfile : boolean
        Flag for creating an external species file. If True, external file 
        specified in variable SPECIESFILENAME will be created and overwritten 
        if SPECIESFILE=1 and DATABASE is not an empty string and COMPONENT is 
        not an empty list.
    spcfileheading : string
        optional string to add as first line in the external species file 
        (appends to 'Pcp_File' string)            
        

    Attributes
    ----------

    Methods
    -------

    See Also
    --------

    Notes
    -----

    Examples
    --------

    >>> import flopy
    >>> m = flopy.mthp.Mthp()
    >>> mcp = flopy.mthp.MthMcp(m)

    """

    def __init__(
        self,
        model,
        speciesfile = 0,
        globalfile = 0,
        nhpfile = 0,
        ftemp = 0,
        basenr = -1,
        speciesfilename = None,
        database = "PHREEQC.DAT",
        component = [],
        globalfilename = None,
        gwfile = "groundwater.phr",
        hpfile = [],
        outputpath = ".",
        ichemdistr = 1,
        temperature = 25.0,
        extension="mcp",
        unitnumber=None,
        filenames=None,
        createspcfile=False,
        spcfileheading=None,
        **kwargs
    ):
        if unitnumber is None:
            unitnumber = MthMcp._defaultunit()
        elif unitnumber == 0:
            unitnumber = MthMcp._reservedunit()

        # call base package constructor
        super().__init__(
            model,
            extension=extension,
            name=self._ftype(),
            unit_number=unitnumber,
            filenames=self._prepare_filenames(filenames),
        )

        # Set dimensions
        nrow = model.nrow
        ncol = model.ncol
        nlay = model.nlay
        
        #Assignements
        self.speciesfile = speciesfile
        self.globalfile = globalfile
        self.nhpfile = nhpfile
        self.ftemp = ftemp
        self.basenr = basenr
        self.speciesfilename = speciesfilename
        self.database = database
        self.component = component
        self.globalfilename = globalfilename
        self.gwfile = gwfile
        self.hpfile = hpfile
        self.outputpath = outputpath
        self.ichemdistr = Util3d(
            model,
            (nlay, nrow, ncol),
            np.int32,
            ichemdistr,
            name="ichemdistr",
            locat=self.unit_number[0],
            array_free_format=False,
        )
        self.temperature = temperature
        self.createspcfile = createspcfile
        self.spcfileheading = spcfileheading
 
        if len(list(kwargs.keys())) > 0:
           raise Exception(
               "MCP error. unrecognized kwargs: "
               + " ".join(list(kwargs.keys()))
           )
 
        #check if the input is consistent
        if self.speciesfile == 0 or self.createspcfile: ## Database & components need to be specified
            if self.database == None:
                s = "MCP error. Database file not specified!"
                raise Exception(s)
            if not(os.path.isfile(self.database)):
                warnings.warn(
                    "MCP warning. Database path (DATABASE) does not exists: "
                    "{}".format(self.database),
                    category = UserWarning)
            if len(self.component) == 0:
                raise Exception(
                    "MCP error. SPECIESFILE = 0 and no chemical components "
                    "specified!") 
        else:
            if not(os.path.exists(os.path.join(model.model_ws,self.speciesfilename))):
                if self.createspcfile:
                    warnings.warn(
                        "MCP warning. File {} will be created".format(self.speciesfilename),
                        category=UserWarning,
                        )
                else:
                    warnings.warn(
                        "MCP warning. File listing the species not found while "
                        "the 'speciesfile=1'. If you want to create the file, "
                        "set the 'createspecfile' to True.",
                        category=UserWarning,
                        )
            else:
                if self.createspcfile:
                    warnings.warn(
                        "MCP warning. File {} will be overwritten!"
                        .format(self.speciesfilename),
                        category=UserWarning,
                        )
                    
        if self.globalfile == 1:
            if not(os.path.exists(os.path.join(model.model_ws,self.globalfilename))):
                warnings.warn(
                    "MCP warning. Global geochemical input file not found "
                    "while the 'globalfile=1'",
                    category=UserWarning,
                    )

        if not(os.path.exists(os.path.join(model.model_ws,self.gwfile))):
            warnings.warn(
                "MCP warning. The path to the Geochemical input file for "
                "MT3D-USGS (GWFILE) does not exist: {}".format(self.gwfile),
                category=UserWarning,
                )

        if self.nhpfile > 0:
            if not(len(self.hpfile) == self.nhpfile):
                raise Exception(
                    "MCP error. The number of Geochemical input files for "
                    "HYDRUS (HPFILE) does not correspond to the number "
                    "specified in NHPFILE!"
                    )
            for hpf in self.hpfile:
                if not(os.path.exists(os.path.join(model.model_ws,hpf))):
                    warnings.warn(
                        "MCP warning. Geochemical input file for HYDRUS '{}'"
                        " not found!".format(hpf),
                        category=UserWarning,                        
                        )
        
        if not(os.path.isdir(os.path.join(model.model_ws,self.outputpath))):
            warnings.warn(
                "MCP warning. OUTPUTPATH {} not found!".format(self.outputpath),
                category=UserWarning,
                )

        # Add self to parent and return
        self.parent.add_package(self)
        return
        

    def write_file(self):
        """
        Write the package file

        Returns
        -------
        None

        """
        # Open file for writing
        f_mcp = open(self.fn_path, "w")
        
        # Record 1
        f_mcp.write("{:10d}{:10d}{:10d}{:10d}{:10d}".format(
            self.speciesfile,
            self.globalfile,
            self.nhpfile,
            self.ftemp,
            self.basenr))
        f_mcp.write("     #SPECIESFILE, GLOBALFILE, NHPFILE, FTEMP, BASENR\n")
        
        # Record 2
        if self.speciesfile == 1:
            f_mcp.write(self.speciesfilename + 
                        "     #SPECIESFILENAME\n")
        
        # Record 3 & 4 (if species are defined internally - SPECIESFILE=0)
        if self.speciesfile == 0:
            if not(os.path.isabs(self.database)):
                f_mcp.write("relative\n")
            f_mcp.write(os.path.normpath(self.database) + "   #DATABASE\n")
            for comp in self.component:
                f_mcp.write(comp + "\n")
        
        # Record 5
        if self.globalfile != 0:
            f_mcp.write(os.path.normpath(self.globalfilename) + 
                        "    #GLOBAL\n")

        # Record 6
        f_mcp.write(self.gwfile + "    #GWFILE\n")
        
        # Record 7
        if self.nhpfile > 0:
            for nhp in self.hpfile:
                f_mcp.write(os.path.normpath(nhp) + "    #HPFILE\n")
        
        # Record 8
        f_mcp.write(os.path.normpath(self.outputpath) + "    #OUTPUTPATH\n")
        
        # Record 9
        f_mcp.write(self.ichemdistr.get_file_entry())
        
        # Record 10
        f_mcp.write("{:10.3f}     #TEMPERATURE\n".format(self.temperature))
        
        f_mcp.close()

        #Optionally write the species file
        

        return

    @classmethod
    def load(
        cls,
        f,
        model,
        nlay=None,
        nrow=None,
        ncol=None,
        nper=None,
        ext_unit_dict=None
    ):
        """
        Load an existing package.

        Parameters
        ----------
        f : filename or file handle
            File to load.
        model : model object
            The model object (of type :class:`flopy.mthp.mth.Mthp`) to
            which this package will be added.
        ext_unit_dict : dictionary, optional
            If the arrays in the file are specified using EXTERNAL,
            or older style array control records, then `f` should be a file
            handle.  In this case ext_unit_dict is required, which can be
            constructed using the function
            :class:`flopy.utils.mfreadnam.parsenamefile`.

        Returns
        -------
        mcp :  MthMcp object
            MthMcp object.

        Examples
        --------

        >>> import flopy
        >>> mth = flopy.mthp.Mthp()
        >>> ssm = flopy.mthp.MthpMcp.load('test.mcp', mth)

        """

        if model.verbose:
            print("loading mcp package file...")

        # Open file, if necessary
        openfile = not hasattr(f, "read")
        if openfile:
            filename = f
            f = open(filename, "r")

        # Set modflow model and dimensions if necessary
        mf = model.mf
        if nlay is None:
            nlay = model.nlay
        if nrow is None:
            nrow = model.nrow
        if ncol is None:
            ncol = model.ncol
        if nper is None:
            nper = model.nper

        # Item D1: SPECIESFILE, GLOBALFILE, NHPFILE, FTEMP, BASENR
        if model.verbose:
            print(
                "   loading SPECIESFILE, GLOBALFILE, NHPFILE, FTEMP, BASENR ..."
            )
        line = f.readline()
        speciesfile = int(line[0:10])
        globalfile = int(line[10:20])
        nhpfile = int(line[20:30])
        ftemp = int(line[30:40])
        basenr = int(line[40:50])
 
        # Item D2: SPECIESFILENAME (if SPECIESFILE = 1)
        if speciesfile != 0:
            if model.verbose:
                print("   loading SPECIESFILENAME...")
            speciesfilename = f.readline().partition("#")[0].strip()
            database = None
            component = []
        
        # Item D3 & D4: DATABASE and COMPONENT (if SPECIESFILE = 0)
        else:
            speciesfilename = None
            if model.verbose:
                print("   loading DATABASE...")
            database = f.readline().partition("#")[0].strip()
            if database == "relative":
                database = f.readline().partition("#")[0].strip()
            if model.verbose:
                print("   loading COMPONENT (MCCOMP={})...".format(model.mccomp))
            if model.mccomp > 0:
                component=[]
                for comp in range(model.mccomp):
                    component.append(f.readline().partition("#")[0].strip())

        # Item D5: GLOBAL (if GLOBALFILE = 1)
        if globalfile != 0:
            if model.verbose:
                print("   loading GLOBAL...")
            globalfilename = f.readline().partition("#")[0].strip()
        else:
            globalfilename = None
        
        # Item D6: GWFILE
        if model.verbose:
            print("   loading GWFILE...")
        gwfile = f.readline().partition("#")[0].strip()
        
        # Item D7: HPFILE (NHPFILE times)
        if nhpfile > 0:
            if model.verbose:
                print("   loading HPFILE (NHPFILE)...")
            hpfile=[]
            for nhp in range(nhpfile):
                hpfile.append(f.readline().partition("#")[0].strip())
        else:
            hpfile=[]
        
        # Item D8: OUTPUTPATH
        if model.verbose:
            print("   loading OUTPUTPATH...")
        outputpath = f.readline().partition("#")[0].strip()
        
        # Item D9: ICHEMDISTR
        if model.verbose:
            print("   loading ICHEMDISTR...")

        ichemdistr = Util3d.load(
            f,
            model,
            (nlay, nrow, ncol),
            int,
            "ichemdistr",
            ext_unit_dict,
            array_format="mt3d",
        )
        
        # Item 10: TEMPERATURE
        if model.verbose:
            print("   loading TEMPERATURE...")
        temperature = float(f.readline()[0:10])

        if openfile:
            f.close()

        # set package unit number
        unitnumber = None
        filenames = [None]
        if ext_unit_dict is not None:
            unitnumber, filenames[0] = model.get_ext_dict_attr(
                ext_unit_dict, filetype=MthMcp._ftype()
            )

        # Construct and return mcp package
        return cls(
            model,
            speciesfile = speciesfile,
            globalfile = globalfile,
            nhpfile = nhpfile,
            ftemp = ftemp,
            basenr = basenr,
            speciesfilename = speciesfilename,
            database = database,
            component = component,
            globalfilename = globalfilename,
            gwfile = gwfile,
            hpfile = hpfile,
            outputpath = outputpath,
            ichemdistr = ichemdistr.array,
            temperature = temperature,
            unitnumber=unitnumber,
            filenames=filenames,
        )

    @staticmethod
    def _ftype():
        return "MCP"

    @staticmethod
    def _defaultunit():
        return 50

    @staticmethod
    def _reservedunit():
        return 5
