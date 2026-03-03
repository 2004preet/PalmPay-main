import React from 'react';
import { TouchableOpacity, View, Text, StyleSheet } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { theme } from '../theme/theme';

export default function ActionIcon({
    icon,
    label,
    onPress,
    color = theme.colors.primary,
    badge,
    isSmall = false
}) {
    return (
        <TouchableOpacity style={styles.container} onPress={onPress} activeOpacity={0.7}>
            <View style={[styles.iconBox, isSmall && styles.iconBoxSmall]}>
                <MaterialCommunityIcons
                    name={icon}
                    size={isSmall ? 22 : 28}
                    color={color}
                />
                {badge && (
                    <View style={styles.badge}>
                        <Text style={styles.badgeText}>{badge}</Text>
                    </View>
                )}
            </View>
            <Text style={styles.label} numberOfLines={2} ellipsizeMode={'tail'}>
                {label}
            </Text>
        </TouchableOpacity>
    );
}

const styles = StyleSheet.create({
    container: {
        alignItems: 'center',
        width: '24%', // Flex into 4 columns per row
        marginVertical: theme.spacing.sm,
    },
    iconBox: {
        width: 56,
        height: 56,
        borderRadius: theme.radius.full,
        backgroundColor: theme.colors.iconBg,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: theme.spacing.xs,
        // Soft shadow for depth
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.05,
        shadowRadius: 2,
        elevation: 2,
    },
    iconBoxSmall: {
        width: 48,
        height: 48,
    },
    label: {
        fontSize: 12,
        color: theme.colors.text,
        textAlign: 'center',
        fontWeight: '500',
        lineHeight: 16,
    },
    badge: {
        position: 'absolute',
        top: -4,
        right: -4,
        backgroundColor: '#ef4444',
        paddingHorizontal: 6,
        paddingVertical: 2,
        borderRadius: 10,
        borderWidth: 2,
        borderColor: theme.colors.surface,
    },
    badgeText: {
        color: '#fff',
        fontSize: 9,
        fontWeight: 'bold',
    }
});
