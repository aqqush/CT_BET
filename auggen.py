"""
AugGen
======
This module implements the AugmentationGenerator class which can be used to
create augmentations of images and their segmentations by performing
affine transformations.
"""
import numpy as np
from scipy import ndimage
import nibabel.orientations as orx


class AugmentationGenerator(object):
    """Initializes an augmentation generator object.

    ### Args
        `rotation_z` (int): Max angle of rotation around z-axis in degrees.
            Defaults to 15.
        `rotation_x` (int): Max angle of rotation around x-axis in degrees.
            Defaults to 2.
        `rotation_y` (int): Max angle of rotation around y-axis in degrees.
            Defaults to 2.
        `translation_xy` (int): Max number of pixels to translate in the
            xy plane. Defaults to 15.
        `translation_z` (int): Max number of pixels to translate in the
            z axis. Defaults to 0.
        `scale_xy` (float): Scaling factor is 1 +/- `scale_delta_max`,
            Defaults to 0.1.
        `scale_z` (int): Scaling factor is 1 +/- `scale_delta_max`,
            Defaults to 0.
        `thickness` (array-like): Slice thickness to pick from. Defaults to (1,)
        `flip_h` (bool): Determines whether horizontal flipping can occur.
            Defaults to True.
        `flip_v` (bool): Determines whether vertical flipping can occur.
            Defaults to False.
        `filling_mode` (str): mode passed to ndimage.affine_transform function,
            ('constant', 'nearest', 'reflect', 'mirror', 'wrap). Defaults to 
            'constant
        `padding_value` (float):  Determines the value used for filling if
            filling mode is 'constant
        `bone_edges` (bool): Determines whether mask pixels are adjusted
            to account for bone edges. Defaults to False.
    """
    def __init__(self, rotation_z=0, rotation_x=0, rotation_y=0,
                 translation_xy=0, translation_z=0, scale_xy=0,
                 scale_z=0, thickness=(0,), flip_h=False, flip_v=False,
                 filling_mode='constant', padding_value=0.0, bone_edges=False, mask_PV=0):
        self.rotation_z = rotation_z
        self.rotation_x = rotation_x
        self.rotation_y = rotation_y
        self.translation_xy = translation_xy
        self.translation_z = translation_z
        self.scale_xy = scale_xy
        self.scale_z = scale_z
        self.thickness = thickness
        self.flip_h = bool(flip_h)
        self.flip_v = bool(flip_v)
        self.filling_mode = filling_mode
        self.padding_value = padding_value
        self.mask_PV = mask_PV
        self.bone_edges = bone_edges

    def __repr__(self):
        string = 'AugmentationGenerator('
        string += 'rotation_z: ' + str(self.rotation_z) + ', '
        string += 'rotation_x: ' + str(self.rotation_x) + ', '
        string += 'rotation_y: ' + str(self.rotation_y) + ', '
        string += 'translation_xy: ' + str(self.translation_xy) + ', '
        string += 'translation_z: ' + str(self.translation_z) + ', '
        string += 'scale_xy: ' + str(self.scale_xy) + ', '
        string += 'scale_z: ' + str(self.scale_z) + ', '
        string += 'thickness: ' + str(self.thickness) + ', '
        string += 'flip_h: ' + str(self.flip_h) + ', '
        string += 'flip_v: ' + str(self.flip_v) + ')'
        string += 'filling_mode: ' + str(self.filling_mode) + ')'
        string += 'padding_value: ' + str(self.padding_value) + ')'
        string += 'bone_edges: ' + str(self.bone_edges) + ')'

        return string

    def __str__(self):
        string = 'ImageAugmentationGenerator parameters:\n'
        string += '  rotation_z: ' + str(self.rotation_z) + '\n'
        string += '  rotation_x: ' + str(self.rotation_x) + '\n'
        string += '  rotation_y: ' + str(self.rotation_y) + '\n'
        string += '  translation_xy: ' + str(self.translation_xy) + '\n'
        string += '  translation_z: ' + str(self.translation_z) + '\n'
        string += '  scale_xy: ' + str(self.scale_xy) + '\n'
        string += '  scale_z: ' + str(self.scale_z) + '\n'
        string += '  thickness: ' + str(self.thickness) + '\n'
        string += '  flip_h: ' + str(self.flip_h) + '\n'
        string += '  flip_v: ' + str(self.flip_v) + '\n'
        string += '  filling_mode: ' + str(self.filling_mode) + '\n'
        string += '  padding_value: ' + str(self.padding_value) + '\n'
        string += '  bone_edges: ' + str(self.bone_edges) + '\n'

        return string

    def _create_comp_affine(self, affine, offset):
        """Returns a composite matrix from the affine and offset."""
        if affine.shape != (2, 2) and affine.shape != (3, 3):
            raise ValueError('Affine must be a 2x2 or 3x3 matrix.')
        if affine.shape[0] != offset.shape[0]:
            raise ValueError('Affine and offset have incompatible dimensions.')

        affine = np.concatenate((affine, offset.reshape(-1, 1)), axis=1)

        if affine.shape == (3, 4):
            affine = np.concatenate((affine, np.array([[0, 0, 0, 1]])), axis=0)
        else:
            affine = np.concatenate((affine, np.array([[0, 0, 1]])), axis=0)

        return affine

    def generate(self, src, mask, n, affine, return_orig=True, verbose=0):
        """Generate `n` augmentations of the `src` and `mask`.

        ### Args
            `src` (three dimensional ndarray): Image matrix.
            `mask` (three dimensional ndarray): Binary mask matrix.
            `n` (int): Number of augmentations.
            `return_orig` (bool): Determines whether original src and mask are
                returned. Defaults to True.
            `verbose` (int): Determines if augmentation parameters are printed
                to screen. Defaults to 0.

        ### Returns:
            Tuple of two matrices, first matrix contains stacked `src`
            augmentations while the second contains the stacked `mask`
            augmentations.
        """
        # Check if src and mask are compatible
        if src.ndim != mask.ndim:
            raise ValueError('Src and mask dimensions don\'t match.')

        # If the passed src is 2 dimensional, call the generate2d method instead
        if src.ndim == 2:
            return self._generate2d(src, mask, n, return_orig, verbose)

#        (x1,y1,z1) = orx.aff2axcodes(affine)
#        ornt = orx.axcodes2ornt((x1,y1,z1))  
#        refOrnt = orx.axcodes2ornt(('R','A','S'))
#        newOrnt1 = orx.ornt_transform(ornt,refOrnt)
        
#        (x2,y2,z2) = orx.aff2axcodes(affine)
#        ornt = orx.axcodes2ornt((x2,y2,z2))  
#        refOrnt = orx.axcodes2ornt(('R','A','S'))
#        newOrnt2 = orx.ornt_transform(ornt,refOrnt)
        
#        src = orx.apply_orientation(src,newOrnt1)
#        mask = orx.apply_orientation(mask,newOrnt1)

#        src = np.fliplr(np.rot90(src,1))
#        mask = np.fliplr(np.rot90(mask,1))

        
        # Get parameters from self
        rotation_z = self.rotation_z
        rotation_x = self.rotation_x
        rotation_y = self.rotation_y
        translation_xy = self.translation_xy
        translation_z = self.translation_z
        scale_xy = self.scale_xy
        scale_z = self.scale_z
        thickness=self.thickness
        flip_h = self.flip_h
        flip_v = self.flip_v
        filling_mode = self.filling_mode
        padding_value = self.padding_value
        bone_edges = self.bone_edges
        mask_PV = self.mask_PV
        # Add original images and mask to output stack if warranted
        if return_orig:
            output_img_stack = src.copy()
            output_mask_stack = mask.copy().astype(float)

        if verbose:
            print(self)

        for iteration in range(n):
            # Create base affine
            affine_w_offset = np.identity(4)

            #----------------------------------------------------#
            # Rotate around z-axis (rotation in the axial plane) #
            #----------------------------------------------------#
            if rotation_z:
                # Choose an angle at random and convert it to radians
                rot_z_angle = np.random.randint(-rotation_z, rotation_z + 1)
                rot_z_angle = np.deg2rad(rot_z_angle)

                affine = \
                    np.array([[np.cos(rot_z_angle), -np.sin(rot_z_angle), 0],
                             [np.sin(rot_z_angle), np.cos(rot_z_angle), 0],
                             [0, 0, 1]])

                # Calculate center offset
                center_input = 0.5 * np.array(src.shape)
                center_output = center_input.dot(affine)
                offset = center_input - center_output

                # Calculate new composite affine
                new_affine_w_offset = self._create_comp_affine(affine, offset)
                affine_w_offset = np.dot(affine_w_offset, new_affine_w_offset)

            #-------------------------------------------------------#
            # Rotate around x-axis (rotation in the sagittal plane) #
            #-------------------------------------------------------#
            if rotation_x:
                # Choose an angle at random and convert it to radians
                rot_x_angle = np.random.randint(-rotation_x, rotation_x + 1)
                rot_x_angle = np.deg2rad(rot_x_angle)

                affine = \
                    np.array([[1, 0, 0],
                             [0, np.cos(rot_x_angle), -np.sin(rot_x_angle)],
                             [0, np.sin(rot_x_angle), np.cos(rot_x_angle)]])

                # Calculate center offset
                center_input = 0.5 * np.array(src.shape)
                center_output = center_input.dot(affine)
                offset = center_input - center_output

                # Calculate new composite affine
                new_affine_w_offset = self._create_comp_affine(affine, offset)
                affine_w_offset = np.dot(affine_w_offset, new_affine_w_offset)

            #------------------------------------------------------#
            # Rotate around y-axis (rotation in the coronal plane) #
            #------------------------------------------------------#
            if rotation_y:
                # Choose an angle at random and convert it to radians
                rot_y_angle = np.random.randint(-rotation_y, rotation_y + 1)
                rot_y_angle = np.deg2rad(rot_y_angle)

                affine = \
                    np.array([[np.cos(rot_y_angle), 0, np.sin(rot_y_angle)],
                             [0, 1, 0],
                             [-np.sin(rot_y_angle), 0, np.cos(rot_y_angle)]])

                # Calculate center offset
                center_input = 0.5 * np.array(src.shape)
                center_output = center_input.dot(affine)
                offset = center_input - center_output

                # Calculate new composite affine
                new_affine_w_offset = self._create_comp_affine(affine, offset)
                affine_w_offset = np.dot(affine_w_offset, new_affine_w_offset)

            #----------------------------------------#
            # Scale along x, y axes, possibly z-axis #
            #----------------------------------------#
            if scale_xy:
                scale_factor = 1 + np.random.uniform(-scale_xy, scale_xy)
                affine = np.identity(3) * scale_factor

                # Eliminate scaling in the z-axis (may be preferable with
                # thick slices)
                if not scale_z:
                    affine[2, 2] = 1
                else:
                    affine[2, 2] = scale_z

                # Calculate center offset
                center_input = 0.5 * np.array(src.shape)
                center_output = center_input.dot(affine)
                offset = center_input - center_output

                # Calculate new composite affine
                new_affine_w_offset = self._create_comp_affine(affine, offset)
                affine_w_offset = np.dot(affine_w_offset, new_affine_w_offset)

            #------------------------------------------------#
            # Translate in the x and y axes, possibly z-axis #
            #------------------------------------------------#
            if translation_xy:
                x_offset = np.random.randint(-translation_xy,
                                             translation_xy + 1)
                y_offset = np.random.randint(-translation_xy,
                                             translation_xy + 1)
            else:
                x_offset, y_offset = 0, 0

            if translation_z:
                z_offset = np.random.randint(-translation_z, translation_z + 1)
            else:
                z_offset = 0

            offset = np.array([x_offset, y_offset, z_offset])

            # Add offset to existing affine offset
            affine_w_offset[:3, 3] = affine_w_offset[:3, 3] + offset

            #----------------------------------------#
            # Apply composite affine to src and mask #
            #----------------------------------------#
            output_img = ndimage.affine_transform(
                input=src,
                matrix=affine_w_offset[:3, :3].T,
                offset=affine_w_offset[:3, 3].ravel(),
                mode=filling_mode,
                cval=padding_value)
            output_mask = ndimage.affine_transform(
                input=mask,
                matrix=affine_w_offset[:3, :3].T,
                offset=affine_w_offset[:3, 3].ravel(),
                mode=filling_mode, cval=mask_PV )

            #---------------------#
            # Make thicker slices #
            #---------------------#
#            if len(thickness) > 1 or thickness[0] != 1:
#                th = np.random.choice(thickness)
#
#                temp_output_img = np.zeros(output_img.shape[:2] + (output_img.shape[2] // th,))
#                temp_output_mask = np.zeros(output_img.shape[:2] + (output_img.shape[2] // th,))
#
#                for i, j in enumerate(range(0, output_img.shape[2] // th * th, th)):
#                    temp_output_img[:, :, i] = np.mean(output_img[:, :, j:j+th], axis=2)
#                    temp_output_mask[:, :, i] = np.mean(output_mask[:, :, j:j+th], axis=2)
#                
#                output_img = temp_output_img
#                output_mask = temp_output_mask
#            else:
#                th = 1

            #---------------------------------------#
            # Flip image horizontally or vertically #
            #---------------------------------------#
            if flip_v:
                flip_v_bool = np.random.choice((True, False))
                if flip_v_bool:
                    output_img = np.fliplr(output_img)
                    output_mask = np.fliplr(output_mask)
            if flip_h:
                flip_h_bool = np.random.choice((True, False))
                if flip_h_bool:
                    output_img = np.flipud(output_img)
                    output_mask = np.flipud(output_mask)

            # Print augmentation parameters to screen
#            if verbose:
#                print(f'Aug {iteration+1}:')
#                if rotation_z:
#                    print('  Rotation about z axis:',
#                          f'{np.rad2deg(rot_z_angle):3} degrees')
#                if rotation_x:
#                    print('  Rotation about x-axis:',
#                          f'{np.rad2deg(rot_x_angle):3} degrees')
#                if rotation_y:
#                    print('  Rotation about y-axis:',
#                          f'{np.rad2deg(rot_y_angle):3} degrees')
#                if scale_xy:
#                    print(f'  Scaling in the x and y axes: {scale_factor:3}')
#                if scale_z:
#                    print(f'  Scaling in the z-axis: {scale_factor:3}')
#                if translation_xy:
#                    print(f'  X offset: {x_offset:3}')
#                if translation_z:
#                    print(f'  Y offset: {y_offset:3}')
#                if th != 1:
#                    print(f'  Slice thickness: {th}')
#                if flip_h:
#                    print(f'  Horizontal flip: {flip_h_bool}')
#                if flip_v:
#                    print(f'  Vertical flip: {flip_v_bool}')
#                print('')

            # Adjust for mask pixels that overlie bone
            if bone_edges:
                output_mask[np.logical_and(output_img > 100, output_mask < 0.85)] = 0

            # Make output_mask a binary mask again
            output_mask = (output_mask > 0.65).astype(float)

            # Concatenate augmentation with remainder of stack
            try:
                output_img_stack = np.concatenate(
                    [output_img_stack, output_img], axis=2)
                output_mask_stack = np.concatenate(
                    [output_mask_stack, output_mask], axis=2)
            except NameError:
                output_img_stack = output_img
                output_mask_stack = output_mask

#        if verbose:
#            print(f'Generated {n} augmentations(s) with a resulting' +
#                  f' stack size of {output_img_stack.shape}.')

        return output_img_stack, output_mask_stack

    def _generate2d(self, src, mask, n, return_orig=True, verbose=0):
        """Generate `n` augmentations of a 2d `src` and `mask`.

        ### Args
            `src` (two dimensional ndarray): Image matrix.
            `mask` (two dimensional ndarray): Binary mask matrix.
            `n` (int): Number of augmentations.
            `return_orig` (bool): Determines whether original src and mask are
                returned. Defaults to True.
            `verbose` (int): Determines if augmentation parameters are printed
                to screen. Defaults to 0.

        ### Returns:
            Tuple of two matrices, first matrix contains stacked `src`
            augmentations while the second contains the stacked `mask`
            augmentations.
        """

        # Get parameters from self
        rotation_z = self.rotation_z
        rotation_x = self.rotation_x
        rotation_y = self.rotation_y
        translation_xy = self.translation_xy
        translation_z = self.translation_z
        scale_xy = self.scale_xy
        scale_z = self.scale_z
        flip_h = self.flip_h
        flip_v = self.flip_v
        filling_mode = self.filling_mode
        padding_value = self.padding_value
        mask_PV = self.mask_PV
        # Add original images and mask to output stack if warranted
        if return_orig:
            output_img_stack = src.reshape(src.shape + (1,)).copy()
            output_mask_stack = mask.reshape(mask.shape + (1,)).copy().astype(float)

        if verbose:
            print(self)
        
        for i in range(n):
            # Create base affine
            affine_w_offset = np.identity(3)

            #----------------------------------------------------#
            # Rotate around z-axis (rotation in the axial plane) #
            #----------------------------------------------------#
            if rotation_z:
                # Choose an angle at random and convert it to radians
                rot_z_angle = np.random.randint(-rotation_z, rotation_z + 1)
                rot_z_angle = np.deg2rad(rot_z_angle)

                affine = \
                    np.array([[np.cos(rot_z_angle), -np.sin(rot_z_angle)],
                             [np.sin(rot_z_angle), np.cos(rot_z_angle)]])

                # Calculate center offset
                center_input = 0.5 * np.array(src.shape)
                center_output = center_input.dot(affine)
                offset = center_input - center_output

                # Calculate new composite affine
                new_affine_w_offset = self._create_comp_affine(affine, offset)
                affine_w_offset = np.dot(affine_w_offset, new_affine_w_offset)

            #-----------------------#
            # Scale along x, y axes #
            #-----------------------#
            if scale_xy:
                scale_factor = 1 + np.random.uniform(-scale_xy, scale_xy)
                affine = np.identity(2) * scale_factor

                # Calculate center offset
                center_input = 0.5 * np.array(src.shape)
                center_output = center_input.dot(affine)
                offset = center_input - center_output

                # Calculate new composite affine
                new_affine_w_offset = self._create_comp_affine(affine, offset)
                affine_w_offset = np.dot(affine_w_offset, new_affine_w_offset)

            #------------------------------------------------#
            # Translate in the x and y axes, possibly z-axis #
            #------------------------------------------------#
            if translation_xy:
                x_offset = np.random.randint(-translation_xy,
                                             translation_xy + 1)
                y_offset = np.random.randint(-translation_xy,
                                             translation_xy + 1)
            else:
                x_offset, y_offset = 0, 0

            offset = np.array([x_offset, y_offset])

            # Add offset to existing affine offset
            affine_w_offset[:2, 2] = affine_w_offset[:2, 2] + offset

            #----------------------------------------#
            # Apply composite affine to src and mask #
            #----------------------------------------#
            output_img = ndimage.affine_transform(
                input=src,
                matrix=affine_w_offset[:2, :2].T,
                offset=affine_w_offset[:2, 2].ravel(),
                mode=filling_mode,
                cval=padding_value)
            output_mask = ndimage.affine_transform(
                input=mask,
                matrix=affine_w_offset[:2, :2].T,
                offset=affine_w_offset[:2, 2].ravel(),
                mode=filling_mode)

            #---------------------------------------#
            # Flip image horizontally or vertically #
            #---------------------------------------#
            if flip_h:
                flip_h_bool = np.random.choice((True, False))
                if flip_h_bool:
                    output_img = np.fliplr(output_img)
                    output_mask = np.fliplr(output_mask)
            if flip_v:
                flip_v_bool = np.random.choice((True, False))
                if flip_v_bool:
                    output_img = np.flipud(output_img)
                    output_mask = np.flipud(output_mask)

            # Print augmentation parameters to screen
#            if verbose:
#                print(f'Aug {i+1}:')
#                if rotation_z:
#                    print('  Rotation about z axis:',
#                          f'{np.rad2deg(rot_z_angle):3} degrees')
#                if rotation_x:
#                    print('  Rotation about x-axis: 2d data, not performed')
#                if rotation_y:
#                    print('  Rotation about y-axis: 2d data, not performed')
#                if scale_xy:
#                    print(f'  Scaling in the x and y axes: {scale_factor:3}')
#                if scale_z:
#                    print('  Scaling in the z-axis: 2d data, not performed.')
#                if translation_xy:
#                    print(f'  X offset: {x_offset:3}')
#                if translation_z:
#                    print('  Y offset: 2d data, not performed.')
#                if flip_h:
#                    print(f'  Horizontal flip: {flip_h_bool}')
#                if flip_v:
#                    print(f'  Vertical flip: {flip_v_bool}')
#                print('')

            # Make output_mask a binary mask again
            output_mask = (output_mask > 0.5).astype(int)

            # Convert output_img and output_mask to 3d matrices for concatenation
            output_img = output_img.reshape(output_img.shape + (1,))
            output_mask = output_mask.reshape(output_mask.shape + (1,))

            # Concatenate augmentation with remainder of stack
            try:
                output_img_stack = np.concatenate(
                    [output_img_stack, output_img], axis=2)
                output_mask_stack = np.concatenate(
                    [output_mask_stack, output_mask], axis=2)
            except NameError:
                output_img_stack = output_img
                output_mask_stack = (output_mask > 0.5).astype(int)

#        if verbose:
#            print(f'Generated {n} 2d augmentations(s) with a resulting' +
#                  f' stack size of {output_img_stack.shape}.')

        return output_img_stack, output_mask_stack
