import React from 'react';
import { View, StyleSheet, ScrollView, StatusBar } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { theme } from '../theme/theme';
import SectionContainer from '../components/SectionContainer';
import ActionIcon from '../components/ActionIcon';
import Header from '../components/Header'; // We will build this next

export default function HomeScreen() {
    return (
        <SafeAreaView style={styles.safeArea} edges={['top']}>
            <StatusBar backgroundColor={theme.colors.primary} barStyle="light-content" />

            {/* PhonePe Signature Purple Header */}
            <Header />

            <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>

                {/* Money Transfers Section (PalmPay Core) */}
                <SectionContainer title="Transactions">
                    <ActionIcon icon="swap-horizontal" label="Transfer" color={theme.colors.secondary} />
                    <ActionIcon icon="arrow-down-bold-circle-outline" label="Deposit" color={theme.colors.secondary} />
                    <ActionIcon icon="arrow-up-bold-circle-outline" label="Withdraw" color={theme.colors.secondary} />
                    <ActionIcon icon="wallet" label="Check Balance" color={theme.colors.secondary} />
                </SectionContainer>

                {/* Quick Links Section */}
                <SectionContainer>
                    <ActionIcon icon="history" label="History" color={theme.colors.blue} isSmall />
                    <ActionIcon icon="hand-wave" label="Biometrics" color={theme.colors.gold} badge="Secure" isSmall />
                    <ActionIcon icon="shield-check" label="Security" color={theme.colors.secondary} isSmall />
                </SectionContainer>

                {/* Account & Administration */}
                <SectionContainer title="Account & Admin">
                    <ActionIcon icon="account-plus" label="Register New" color={theme.colors.blue} />
                    <ActionIcon icon="account-group" label="View Users" color={theme.colors.blue} />
                    <ActionIcon icon="brain" label="Retrain AI" color={theme.colors.gold} />
                    <ActionIcon icon="cog" label="Settings" color={theme.colors.muted} />
                </SectionContainer>

            </ScrollView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    safeArea: {
        flex: 1,
        backgroundColor: theme.colors.primary, // The notch area matches the header
    },
    scrollContent: {
        backgroundColor: theme.colors.background,
        paddingTop: theme.spacing.md,
        paddingBottom: theme.spacing.xl,
        maxWidth: 600, // Constrain web width
        width: '100%',
        alignSelf: 'center'
    }
});
