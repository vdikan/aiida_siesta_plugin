# -*- coding: utf-8 -*-
import os

from aiida.common.constants import elements
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.exceptions import InputValidationError
from aiida.common.utils import classproperty
from aiida.orm.calculation.job import JobCalculation
from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.remote import RemoteData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.singlefile import SinglefileData

# Module with fdf-aware dictionary
from tkdict import FDFDict

__copyright__ = u"Copyright (c), 2015, ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE (Theory and Simulation of Materials (THEOS) and National Centre for Computational Design and Discovery of Novel Materials (NCCR MARVEL)), Switzerland and ROBERT BOSCH LLC, USA. All rights reserved."
__license__ = "MIT license, see LICENSE.txt file"
__version__ = "0.12.0"
__contributors__ = "Victor M. Garcia-Suarez, ..."

class VibraCalculation(JobCalculation):
    """
    Plugin for the Vibra program in the Siesta-Vibra distribution, which
    computes the phonon spectrum from a Siesta calculation.
    """
    _siesta_plugin_version = 'aiida-0.11.0--plugin-0.11.5'

    def _init_internal_params(self):
        super(VibraCalculation, self)._init_internal_params()

        # Default Siesta output parser provided by AiiDA
        self._default_parser = "siesta.vibra"

        # Keywords that cannot be set
        # We need to canonicalize this!

        self._aiida_blocked_keywords = ['system-name', 'system-label']

        self._aiida_blocked_keywords.append('number-of-atoms')
        #
        #
        self._aiida_blocked_keywords.append('latticeconstant')
        self._aiida_blocked_keywords.append('lattice-constant')
        self._aiida_blocked_keywords.append('atomic-coordinates-format')
        self._aiida_blocked_keywords.append('atomiccoordinatesformat')

        # Default input and output files
        self._DEFAULT_INPUT_FILE = 'aiida.in'
        self._DEFAULT_OUTPUT_FILE = 'aiida.out'
        self._DEFAULT_FC_FILE = 'aiida.FC'
        self._DEFAULT_BANDS_FILE = 'aiida.bands'
        self._DEFAULT_VECTORS_FILE = 'aiida.vectors'

        self._SFILES_SUBFOLDER = './'
        self._OUTPUT_SUBFOLDER = './'
        self._PREFIX = 'aiida'
        self._INPUT_FILE_NAME = 'aiida.fdf'
        self._OUTPUT_FILE_NAME = 'aiida.out'
        self._FC_FILE_NAME = 'aiida.FC'
        self._BANDS_FILE_NAME = 'aiida.bands'
        self._VECTORS_FILE_NAME = 'aiida.vectors'

        # in restarts, it will copy from the parent the following
        self._restart_copy_from = os.path.join(self._OUTPUT_SUBFOLDER, '*.FC')

        # in restarts, it will copy the previous folder in the following one
        self._restart_copy_to = self._OUTPUT_SUBFOLDER

    @classproperty
    def _use_methods(cls):
        """
        Extend the parent _use_methods with further keys.
        """
        retdict = JobCalculation._use_methods

        retdict["structure"] = {
            'valid_types': StructureData,
            'additional_parameter': None,
            'linkname': 'structure',
            'docstring': "Choose the input structure to use",
        }
        retdict["settings"] = {
            'valid_types': ParameterData,
            'additional_parameter': None,
            'linkname': 'settings',
            'docstring': "Use an additional node for special settings",
        }
        retdict["parameters"] = {
            'valid_types': ParameterData,
            'additional_parameter': None,
            'linkname': 'parameters',
            'docstring': ("Use a node that specifies the input parameters "
                          "for the namelists"),
        }
        retdict['bandskpoints'] = {
            'valid_types': KpointsData,
            'additional_parameter': None,
            'linkname': 'bandskpoints',
            'docstring': ("Use the node defining the kpoint sampling"
                          "to use for bands calculation"),
        }
        retdict["parent_folder"] = {
            'valid_types': RemoteData,
            'additional_parameter': None,
            'linkname': 'parent_calc_folder',
            'docstring': ("Use a remote folder as parent folder (for "
                          "restarts and similar"),
        }
        retdict['singlefile'] = {
            'valid_types': SinglefileData,
            'additional_parameter': None,
            'linkname': 'singlefile',
            'docstring': ("FC file that is needed to run the calculation"),
        }
        return retdict

    def _prepare_for_submission(self, tempfolder, inputdict):
        """
        This is the routine to be called when you want to create
        the input files and related stuff with a plugin.

        :param tempfolder: a aiida.common.folders.Folder subclass where
                           the plugin should put all its files.
        :param inputdict: a dictionary with the input nodes, as they would
                be returned by get_inputdata_dict (without the Code!)
        """

        local_copy_list = []
        remote_copy_list = []

        # Process the settings dictionary first
        # Settings can be undefined, and defaults to an empty dictionary
        settings = inputdict.pop(self.get_linkname('settings'), None)
        if settings is None:
            settings_dict = {}
        else:
            if not isinstance(settings, ParameterData):
                raise InputValidationError(
                    "settings, if specified, must be of "
                    "type ParameterData")

            # Settings converted to UPPERCASE
            # Presumably to standardize the usage and avoid
            # ambiguities
            settings_dict = _uppercase_dict(
                settings.get_dict(), dict_name='settings')

        try:
            parameters = inputdict.pop(self.get_linkname('parameters'))
        except KeyError:
            raise InputValidationError("No parameters specified for this "
                                       "calculation")
        if not isinstance(parameters, ParameterData):
            raise InputValidationError("parameters is not of type "
                                       "ParameterData")

        try:
            structure = inputdict.pop(self.get_linkname('structure'))
        except KeyError:
            raise InputValidationError("No structure specified for this "
                                       "calculation")
        if not isinstance(structure, StructureData):
            raise InputValidationError(
                "structure is not of type StructureData")

        bandskpoints = inputdict.pop(self.get_linkname('bandskpoints'), None)
        if bandskpoints is None:
            flagbands = False
        else:
            flagbands = True
            if not isinstance(bandskpoints, KpointsData):
                raise InputValidationError(
                    "kpoints for bands is not of type KpointsData")

        singlefile = inputdict.pop(self.get_linkname('singlefile'), None)
        if singlefile is not None:
            if not isinstance(singlefile, SinglefileData):
                raise InputValidationError("singlefile, if specified,"
                                           "must be of type SinglefileData")

        parent_calc_folder = inputdict.pop(
            self.get_linkname('parent_folder'), None)
        if parent_calc_folder is not None:
            if not isinstance(parent_calc_folder, RemoteData):
                raise InputValidationError("parent_calc_folder, if specified,"
                                           "must be of type RemoteData")

        try:
            code = inputdict.pop(self.get_linkname('code'))
        except KeyError:
            raise InputValidationError(
                "No code specified for this calculation")

        # Here, there should be no more parameters...
        if inputdict:
            raise InputValidationError(
                "The following input data nodes are "
                "unrecognized: {}".format(inputdict.keys()))

        ##############################
        # END OF INITIAL INPUT CHECK #
        ##############################

        #
        # There should be a warning for duplicated (canonicalized) keys
        # in the original dictionary in the script

        input_params = FDFDict(parameters.get_dict())

        # Look for blocked keywords and
        # add the proper values to the dictionary

        for blocked_key in self._aiida_blocked_keywords:
            canonical_blocked = FDFDict.translate_key(blocked_key)
            for key in input_params:
                if key == canonical_blocked:
                    raise InputValidationError(
                        "You cannot specify explicitly the '{}' flag in the "
                        "input parameters".format(
                            input_params.get_last_key(key)))

        input_params.update({'system-name': self._PREFIX})
        input_params.update({'system-label': self._PREFIX})

        input_params.update({'number-of-species': len(structure.kinds)})
        input_params.update({'number-of-atoms': len(structure.sites)})
        #
        # Regarding the lattice-constant parameter:
        # -- The variable "alat" is not typically kept anywhere, and
        # has already been used to define the vectors.
        # We need to specify that the units of these vectors are Ang...

        input_params.update({'lattice-constant': '1.0 Ang'})

        # Note that this  will break havoc with the band-k-points "pi/a"
        # option. The use of this option should be banned.

        # Note that the implicit coordinate convention of the Structure
        # class corresponds to the "Ang" convention in Siesta.
        # The "atomic-coordinates-format" keyword is blocked to ScaledCartesian,
        # which is given in terms of the lattice constant (1.0 Ang).
        input_params.update({'atomic-coordinates-format': 'ScaledCartesian'})

        # ============== Preparation of input data ===============
        #

        # ------------ CELL_PARAMETERS -----------
        cell_parameters_card = "%block lattice-vectors\n"
        for vector in structure.cell:
            cell_parameters_card += ("{0:18.10f} {1:18.10f} {2:18.10f}"
                                     "\n".format(*vector))
        cell_parameters_card += "%endblock lattice-vectors\n"

        # ------------- ATOMIC_SPECIES ------------
        # Only the species index and the mass are necessary 

        # Dictionary to get the mass of a given element
        datmn = dict([(v['symbol'],v['mass']) for k, v in elements.iteritems()])

        spind = {}
        spcount = 0
        for kind in structure.kinds:

            spcount += 1
            spind[kind.name] = spcount

        # ------------ ATOMIC_POSITIONS -----------
        atomic_positions_card_list = [
            "%block atomiccoordinatesandatomicspecies\n"
        ]
        countatm = 0
        for site in structure.sites:
            countatm += 1
            atomic_positions_card_list.append(
                "{0:18.10f} {1:18.10f} {2:18.10f} {3:4} {4:10} \n".format(
                    site.position[0], site.position[1], site.position[2],
                    spind[site.kind_name], datmn[kind.symbol]))
        atomic_positions_card = "".join(atomic_positions_card_list)
        del atomic_positions_card_list  # Free memory
        atomic_positions_card += "%endblock atomiccoordinatesandatomicspecies\n"

        # --------------- K-POINTS-FOR-BANDS ----------------!
        #This part is computed only if flagbands=True
        #Two possibility are supported in Siesta: BandLines ad BandPoints
        #At the moment the user can't choose directly one of the two options
        #BandsLine is set automatically if bandskpoints has labels,
        #BandsPoints if bandskpoints has no labels
        #BandLinesScale =pi/a is not supported at the moment because currently
        #a=1 always. BandLinesScale ReciprocalLatticeVectors is always set
        if flagbands:
            bandskpoints_card_list = [
                "BandLinesScale ReciprocalLatticeVectors\n"
            ]
            if bandskpoints.labels == None:
                bandskpoints_card_list.append("%block BandPoints\n")
                for s in bandskpoints.get_kpoints():
                    bandskpoints_card_list.append(
                        "{0:8.3f} {1:8.3f} {2:8.3f} \n".format(
                            s[0], s[1], s[2]))
                fbkpoints_card = "".join(bandskpoints_card_list)
                fbkpoints_card += "%endblock BandPoints\n"
            else:
                bandskpoints_card_list.append("%block BandLines\n")
                savs = []
                listforbands = bandskpoints.get_kpoints()
                for s, m in bandskpoints.labels:
                    savs.append(s)
                rawindex = 0
                for s, m in bandskpoints.labels:
                    rawindex = rawindex + 1
                    x, y, z = listforbands[s]
                    if rawindex == 1:
                        bandskpoints_card_list.append(
                            "{0:3} {1:8.3f} {2:8.3f} {3:8.3f} {4:1}\n".format(
                                1, x, y, z, m))
                    else:
                        bandskpoints_card_list.append(
                            "{0:3} {1:8.3f} {2:8.3f} {3:8.3f} {4:1}\n".format(
                                s - savs[rawindex - 2], x, y, z, m))
                fbkpoints_card = "".join(bandskpoints_card_list)
                fbkpoints_card += "%endblock BandLines\n"
            del bandskpoints_card_list

        # -------------ADDITIONAL FILES -----------
        # I create the subfolder that will contain additional Siesta files
        tempfolder.get_subfolder(self._SFILES_SUBFOLDER, create=True)
        # I create the subfolder with the output data
        tempfolder.get_subfolder(self._OUTPUT_SUBFOLDER, create=True)

        if singlefile is not None:
            lfile=singlefile.get_file_abs_path().split("path/",1)[1]
            local_copy_list.append((singlefile.get_file_abs_path(),
                os.path.join(self._SFILES_SUBFOLDER, lfile)))

        # ================ Namelists and cards ===================

        input_filename = tempfolder.get_abs_path(self._INPUT_FILE_NAME)

        with open(input_filename, 'w') as infile:
            # here print keys and values tp file

            for k, v in sorted(input_params.iteritems()):
                infile.write(get_input_data_text(k, v))
                # ,mapping=mapping_species))

            # Write previously generated cards now
            infile.write("#\n# -- Structural Info follows\n#\n")
            infile.write(cell_parameters_card)
            infile.write(atomic_positions_card)
            if flagbands:
                infile.write("#\n# -- Bandlines/Bandpoints Info follows\n#\n")
                infile.write(fbkpoints_card)

        # ------------------------------------- END of fdf file creation

        # The presence of a 'parent_calc_folder' input node signals
        # that we want to get something from there, as indicated in the
        # self._restart_copy_from attribute.
        # In Siesta's case, for now, it is just the density-matrix file
        #
        # It will be copied to the current calculation's working folder.

        if parent_calc_folder is not None:
            remote_copy_list.append(
                (parent_calc_folder.get_computer().uuid, os.path.join(
                    parent_calc_folder.get_remote_path(),
                    self._restart_copy_from), self._restart_copy_to))

        calcinfo = CalcInfo()

        calcinfo.uuid = self.uuid
        #
        # Empty command line by default
        # Why use 'pop' ?
        cmdline_params = settings_dict.pop('CMDLINE', [])

        # Comment this paragraph better, if applicable to Siesta
        #
        #we commented calcinfo.stin_name and added it here in cmdline_params
        #in this way the mpirun ... pw.x ... < aiida.in
        #is replaced by mpirun ... pw.x ... -in aiida.in
        # in the scheduler, _get_run_line, if cmdline_params is empty, it
        # simply uses < calcinfo.stin_name

        if cmdline_params:
            calcinfo.cmdline_params = list(cmdline_params)
        calcinfo.local_copy_list = local_copy_list
        calcinfo.remote_copy_list = remote_copy_list

        calcinfo.stdin_name = self._INPUT_FILE_NAME
        calcinfo.stdout_name = self._OUTPUT_FILE_NAME
        calcinfo.fc_name = self._FC_FILE_NAME

        #
        # Code information object
        #
        codeinfo = CodeInfo()
        codeinfo.cmdline_params = list(cmdline_params)
        codeinfo.stdin_name = self._INPUT_FILE_NAME
        codeinfo.stdout_name = self._OUTPUT_FILE_NAME
        codeinfo.fc_name = self._FC_FILE_NAME
        codeinfo.code_uuid = code.uuid
        calcinfo.codes_info = [codeinfo]

        # Retrieve by default: the output file, the xml file, and the
        # messages file.
        # If flagbands=True we also add the bands file to the retrieve list!
        # This is extremely important because the parser parses the bands
        # only if aiida.bands is in the retrieve list!!

        calcinfo.retrieve_list = []
        calcinfo.retrieve_list.append(self._OUTPUT_FILE_NAME)
        if flagbands:
            calcinfo.retrieve_list.append(self._BANDS_FILE_NAME)

        # Any other files specified in the settings dictionary
        settings_retrieve_list = settings_dict.pop('ADDITIONAL_RETRIEVE_LIST',
                                                   [])
        calcinfo.retrieve_list += settings_retrieve_list

        return calcinfo

    def _set_parent_remotedata(self, remotedata):
        """
        Used to set a parent remotefolder that holds the .FC file
        from a previous Siesta calculation
        """
        from aiida.common.exceptions import ValidationError

        if not isinstance(remotedata, RemoteData):
            raise ValueError('remotedata must be a RemoteData')

        # complain if another remotedata is already found
        input_remote = self.get_inputs(node_type=RemoteData)
        if input_remote:
            raise ValidationError(
                "Cannot set several parent calculation to a "
                "{} calculation".format(self.__class__.__name__))

        self.use_parent_folder(remotedata)


def get_input_data_text(key, val, mapping=None):
    """
    Given a key and a value, return a string (possibly multiline for arrays)
    with the text to be added to the input file.

    :param key: the flag name
    :param val: the flag value. If it is an array, a line for each element
            is produced, with variable indexing starting from 1.
            Each value is formatted using the conv_to_fortran function.
    :param mapping: Optional parameter, must be provided if val is a dictionary.
            It maps each key of the 'val' dictionary to the corresponding
            list index. For instance, if ``key='magn'``,
            ``val = {'Fe': 0.1, 'O': 0.2}`` and ``mapping = {'Fe': 2, 'O': 1}``,
            this function will return the two lines ``magn(1) = 0.2`` and
            ``magn(2) = 0.1``. This parameter is ignored if 'val'
            is not a dictionary.
    """
    from aiida.common.utils import conv_to_fortran
    # I check first the dictionary, because it would also match
    # hasattr(__iter__)
    if isinstance(val, dict):
        if mapping is None:
            raise ValueError("If 'val' is a dictionary, you must provide also "
                             "the 'mapping' parameter")

        list_of_strings = []
        for elemk, itemval in val.iteritems():
            try:
                idx = mapping[elemk]
            except KeyError:
                raise ValueError("Unable to find the key '{}' in the mapping "
                                 "dictionary".format(elemk))

            list_of_strings.append((idx, "  {0}({2}) = {1}\n".format(
                key, conv_to_fortran(itemval), idx)))

        # I first have to resort, then to remove the index from the first
        # column, finally to join the strings
        list_of_strings = zip(*sorted(list_of_strings))[1]
        return "".join(list_of_strings)
    elif hasattr(val, '__iter__'):
        # a list/array/tuple of values
        list_of_strings = [
            "{0}({2})  {1}\n".format(key, conv_to_fortran(itemval), idx + 1)
            for idx, itemval in enumerate(val)
        ]
        return "".join(list_of_strings)
    else:
        # single value
        if key[:6] == '%block':
            bname = key.split()[1]
            b1 = "{0}  {1}".format(key, my_conv_to_fortran(val))
            return b1 + "\n%endblock " + bname + "\n"
        else:
            return "{0}  {1}\n".format(key, my_conv_to_fortran(val))


def my_conv_to_fortran(val):
    """
    Special version to avoid surrounding strings with extra ' '. Otherwise the
    fdf tokenizer will not split values and units, for example.

    :param val: the value to be read and converted to a Fortran-friendly string.
    """
    # Note that bool should come before integer, because a boolean matches also
    # isinstance(...,int)
    if (isinstance(val, bool)):
        if val:
            val_str = '.true.'
        else:
            val_str = '.false.'
    elif (isinstance(val, (int, long))):
        val_str = "{:d}".format(val)
    elif (isinstance(val, float)):
        val_str = ("{:18.10e}".format(val)).replace('e', 'd')
    elif (isinstance(val, basestring)):
        val_str = "{!s}".format(val)
    else:
        raise ValueError("Invalid value passed, accepts only bools, ints, "
                         "floats and strings")

    return val_str


def _uppercase_dict(d, dict_name):
    from collections import Counter

    if isinstance(d, dict):
        new_dict = dict((str(k).upper(), v) for k, v in d.iteritems())
        if len(new_dict) != len(d):

            num_items = Counter(str(k).upper() for k in d.keys())
            double_keys = ",".join([k for k, v in num_items if v > 1])
            raise InputValidationError(
                "Inside the dictionary '{}' there are the following keys that "
                "are repeated more than once when compared case-insensitively: "
                "{}."
                "This is not allowed.".format(dict_name, double_keys))
        return new_dict
    else:
        raise TypeError(
            "_lowercase_dict accepts only dictionaries as argument")
