import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { theme } from '../theme/theme';

export default function Header() {
    return (
        <View style={styles.headerContainer}>
            {/* Top Row: User, Location, Help, QR, Bell */}
            <View style={styles.topRow}>

                {/* Left: User Avatar & Location */}
                <View style={styles.leftSection}>
                    <TouchableOpacity style={styles.avatarContainer}>
                        <Image
                            source={{ uri: 'https://cdn-icons-png.flaticon.com/512/3135/3135715.png' }}
                            style={styles.avatar}
                        />
                    </TouchableOpacity>
                    <View style={styles.locationContainer}>
                        <Text style={styles.locationLabel}>Add Address</Text>
                        <TouchableOpacity style={styles.locationDropdown}>
                            <Text style={styles.locationValue} numberOfLines={1}>Mumbai, India</Text>
                            <MaterialCommunityIcons name="chevron-down" size={20} color="#fff" />
                        </TouchableOpacity>
                    </View>
                </View>

                {/* Right: Icons */}
                <View style={styles.rightSection}>
                    <TouchableOpacity style={styles.iconButton}>
                        <MaterialCommunityIcons name="qrcode-scan" size={22} color="#fff" />
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.iconButton}>
                        <MaterialCommunityIcons name="bell-outline" size={24} color="#fff" />
                    </TouchableOpacity>
                    <TouchableOpacity style={styles.iconButton}>
                        <MaterialCommunityIcons name="help-circle-outline" size={24} color="#fff" />
                    </TouchableOpacity>
                </View>

            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    headerContainer: {
        backgroundColor: theme.colors.primary, // PhonePe Purple
        paddingHorizontal: theme.spacing.md,
        paddingVertical: theme.spacing.md,
        paddingBottom: theme.spacing.xl, // Extended bottom to allow cards to float over
    },
    topRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
    },
    leftSection: {
        flexDirection: 'row',
        alignItems: 'center',
        flex: 1,
    },
    avatarContainer: {
        marginRight: theme.spacing.sm,
    },
    avatar: {
        width: 44,
        height: 44,
        borderRadius: 22,
        borderWidth: 1.5,
        borderColor: '#ffffff50',
    },
    locationContainer: {
        justifyContent: 'center',
        flex: 1,
        marginRight: theme.spacing.md,
        flexShrink: 1, // Prevent squashing
    },
    locationLabel: {
        color: '#ffffff80', // faded white
        fontSize: 12,
        fontWeight: '600',
        marginBottom: 2, // Add spacing so it doesn't overlap
    },
    locationDropdown: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    locationValue: {
        color: '#fff',
        fontSize: 14,
        fontWeight: 'bold',
        marginRight: 2,
        maxWidth: '80%',
        flexShrink: 1, // Allow text to truncate nicely if too long
    },
    rightSection: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 4,
    },
    iconButton: {
        padding: 6,
        borderRadius: theme.radius.full,
        backgroundColor: '#ffffff20', // subtle opaque white circle
        marginLeft: 6,
    }
});
